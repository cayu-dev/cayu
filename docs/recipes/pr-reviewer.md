# Recipe: a cloud PR-review agent

**Goal:** build an agent that, when a pull request opens, checks the queued head
SHA out into a fresh review workspace, QAs the change by running the project's
tests, and posts one review comment back — durably, on a worker pool, not as a
synchronous chat reply.

Every ingredient below is a first-class cayu primitive. The point of this recipe
is how they *compose*. The complete, runnable code is in
[`examples/github_pr_reviewer/`](../../examples/github_pr_reviewer/); run the
no-key demo with:

```bash
PYTHONPATH=src python examples/github_pr_reviewer/pr_reviewer.py
```

The example is split the way a real agent app tends to grow: `reviewer_app.py`
assembles Cayu primitives, `github_tools.py` owns GitHub egress, `workspace.py`
owns checkout setup, `worker.py` owns durable task handling, `webhook.py` owns
external ingress, and `demo.py` owns the deterministic no-key fixture.

## The shape

```
GitHub webhook (pull_request)  ─▶ HMAC-verify ─▶ durable Task (SQLiteTaskStore)
        │                                                     │
  build_webhook_app()                            a worker claims the Task
                                                              ▼
                                          fresh git checkout of the queued PR head SHA
                                              (EnvironmentFactory + binding)
                                                              ▼
                          get_pr_diff ▶ read changed code ▶ run QA commands
                          (restricted to a test/build allowlist) ▶ post_pr_comment
```

Each numbered step maps to one decision.

## 1. Trigger: turn a webhook into a durable Task

An agent that "looks at PRs" is *triggered by* PRs. cayu's `EventWatcher` watches
cayu's **own** durable event log (budget alerts, session completion); it is not an
inbound-webhook receiver. So the ingress is ordinary app code: a small HMAC-verified
endpoint that translates the GitHub payload into a `Task`.

```python
task = await task_store.create_task(TaskCreate(
    type="review_pr",
    assigned_agent_name="pr-reviewer",
    input={"owner": ..., "repo": ..., "pr_number": ..., "repo_url": ...,
           "head_ref": ..., "head_sha": ..., "base_ref": ...},
))
```

Verify the signature with `cayu.webhooks.verify_webhook_signature` (constant-time
HMAC), and derive the `task_id` from the delivery id with
`cayu.webhooks.webhook_task_id("github", delivery_id)` — a redelivered webhook then
maps to a duplicate id the task store rejects, so retries are idempotent for free.
The endpoint returns immediately; the review happens later on a worker — that
decoupling is what makes it "cloud".

## 2. A worker claims the Task and starts a session

Use `run_task_worker(...)` for this shape: it leases matching tasks, keeps the
lease heartbeated while the handler runs, reclaims expired leases, and keeps the
worker alive if one task fails. The handler only has to turn the claimed task
into a fresh `RunRequest`:

```python
async def handle_pr_review_task(app, task, worker_id):
    owner, repo, pr = task.input["owner"], task.input["repo"], task.input["pr_number"]
    short_head_sha = task.input["head_sha"][:12]
    request = RunRequest(
        agent_name="pr-reviewer",
        session_id=f"pr-review-{owner}-{repo}-{pr}-{short_head_sha}",
        task_id=task.id,
        task_worker_id=worker_id,
        environment_name="pr-workspace",
        metadata={"repo_owner": owner, "repo_name": repo, "pr_number": pr,
                  "repo_url": task.input["repo_url"],
                  "head_ref": task.input["head_ref"],
                  "head_sha": task.input["head_sha"],
                  "base_ref": task.input["base_ref"]},
        messages=[Message.text("user", f"Review pull request #{pr} in {owner}/{repo}.")],
        limits=RunLimits(max_tool_calls=20, max_elapsed_seconds=600),
    )
    async for event in app.run(request):
        ...

await run_task_worker(
    app,
    task_store,
    handle_pr_review_task,
    worker_id="worker-1",
    query=TaskQuery(type="review_pr", assigned_agent_name="pr-reviewer"),
    lease_seconds=900,
)
```

Two things worth calling out:

- **`metadata` fans out to everything downstream.** The environment factory reads
  it to know which repo to check out; the tools read it (via `ctx.metadata`) to
  know which PR to fetch and comment on. It is the single source of PR identity.
- **`RunLimits` are your blast-radius cap** — a QA agent runs shell, so bound its
  tool calls and wall-clock. (Limits are enforced per run; see
  [runtime-contracts](../runtime-contracts.md).)

## 3. A fresh review workspace per PR

Register an `EnvironmentFactory` (not a static `Environment`) so every review gets
its own workspace + runner + git checkout, driven by the request metadata:

```python
class PRReviewWorkspaceFactory(EnvironmentFactory):
    async def create(self, request):
        root = self._base_root / request.session_id
        pr_number = request.metadata["pr_number"]
        head_sha = request.metadata["head_sha"]
        checkout_ref = f"refs/cayu/pr-{pr_number}"
        return EnvironmentFactoryResult(environment=Environment(
            EnvironmentSpec(name=request.environment_name),
            workspace=LocalWorkspace(root, workspace_id=request.session_id),
            runner=LocalRunner(root),
            binding=GitRepositoryBinding(
                repo_url=request.metadata["repo_url"],
                ref=head_sha,
                fetch_refspecs=[f"+refs/pull/{pr_number}/head:{checkout_ref}"],
            ),
            vault=vault, proxy=proxy,
        ))
```

- **Same-repo and fork PRs** use the same GitHub pull-ref path:
  fetch `refs/pull/N/head` into a local review ref, then bind `ref=head_sha`.
  This makes the queued commit exact while still supporting fork PR refs.
- **Head movement guard:** `get_pr_diff` validates that GitHub still reports the
  queued `head_sha` before returning diff metadata. If the PR moved, enqueue a new
  review for the new head.
- **Reviewer/QA split:** if you want an orchestrator agent with no checkout and a
  separate QA agent with a checkout, branch the *same* factory on
  `request.agent_name` — it is a per-session field — returning `NoWorkspaceBinding`
  for one and a `GitRepositoryBinding` for the other.
- **Go cloud:** swap `LocalRunner`/`LocalWorkspace` for `E2BRunner`/`E2BWorkspace`
  (`cayu[e2b]`) for real isolation instead of the trusted local-dev runner.

## 4. GitHub egress: two options

cayu ships no GitHub primitive on purpose. Pick either:

**a) A custom `Tool` through the credential proxy** (used in the example). The
token never lives in the tool — it is resolved from the vault at call time:

```python
authorization = await ctx.proxy.authorize_request(destination=url,
    credential=SecretRef(name="github_token"), action="post_pr_comment", metadata={...})
resolved = await ctx.proxy.resolve(SecretRef(name="github_token"),
    scope={"session_id": ctx.session_id, "tool": "post_pr_comment"})
```

If no proxy is configured the tool **fails closed** — it never reaches the network.

**b) The GitHub MCP server** — zero custom tool code:

```python
from cayu import HttpMcpClient, LocalEnvVault, McpServerSpec, SecretRef, connect_mcp_toolset

vault = LocalEnvVault({"github_token": "GITHUB_TOKEN"})
toolset = await connect_mcp_toolset(
    McpServerSpec(
        name="github",
        url="https://api.githubcopilot.com/mcp/",
        secret_headers={"Authorization": SecretRef(name="github_token")},
    ),
    client=HttpMcpClient(secret_resolver=vault),
)
# toolset.tools -> register on the agent; await toolset.close() when done.
```

`secret_headers` are resolved through the HTTP client's secret resolver. For a
self-hosted alternative, point stdio at
[`github/github-mcp-server`](https://github.com/github/github-mcp-server).

## 5. QA safety: constrain `exec_command`

"QA end-to-end" means letting the agent run the project's tests — which is exactly
where you want a guardrail. `ExecCommandTool(policy=...)` takes a `CommandPolicy`:

```python
class QaCommandPolicy(CommandPolicy):
    async def evaluate(self, ctx, request):
        if request.command.kind == "shell":
            return CommandPolicyResult(decision=CommandPolicyDecision.DENY,
                reason="Raw shell strings are not allowed; use kind='process' with argv.")
        if not request.command.argv or request.command.argv[0] not in ALLOWED:
            return CommandPolicyResult(decision=CommandPolicyDecision.DENY, reason=...)
        return CommandPolicyResult(decision=CommandPolicyDecision.ALLOW)
```

A denied command surfaces as a `tool.call.failed` event with a structured error the
model can read and route around — the demo shows a `rm -rf /` attempt being denied.

## 6. The agent

```python
app.register_agent(
    AgentSpec(name="pr-reviewer", model=model, system_prompt=REVIEWER_SYSTEM_PROMPT),
    tools=[GetPRDiffTool(), PostPRCommentTool(), ReadFileTool(), ListFilesTool(),
           ExecCommandTool(policy=QaCommandPolicy())],
    tool_policy=StaticToolPolicy(allow=[...]),
)
```

The system prompt tells the agent the sequence (diff → read → QA → comment once).
`get_pr_diff` is intentionally bounded: it paginates GitHub's changed-file list,
returns explicit `total_files` / truncation flags, and includes only patch previews
under a fixed character budget. The checkout is the source of truth; when a file
needs exact inspection, the agent should use `read_file` against the workspace.
`app.run()` never raises on a model/tool failure — a failed run ends in a terminal
`session.failed` event, so branch on `EventType.SESSION_FAILED` and read the answer
with `final_output_text(transcript)` rather than expecting an exception.

## Going to production

- **Durability:** back the app with `PostgresSessionStore` / `PostgresTaskStore`
  (`pip install "cayu[postgres]"`) so many workers can claim safely.
- **Isolation:** `E2BRunner` / `E2BWorkspace` for the sandbox.
- **Worker loop:** `run_task_worker(...)` instead of a hand-rolled claim loop.
- **Scale:** run N identical workers; the task store's `FOR UPDATE SKIP LOCKED`
  claiming keeps them from colliding.
