# Session fork overlays

A fork overlay lets an application outside the host, such as a supervisor,
fork a host session with its own instructions, its own tools, and its own
memory context. The host app ships no controller code. It enrolls the
controller once, and the controller sends overlays through an authenticated
entrance.

Every part of an overlay is optional, and the parts are independent:

| Part | What it does | Where it lives |
| --- | --- | --- |
| `instructions` | Text the model sees after the copied source history. | Child metadata, rendered per request. |
| `memory` | `ForkOverlayMemoryItem`s visible to this fork only. | Child metadata, rendered per request. |
| `tools` | `ForkOverlayToolGrant`s naming tools the controller serves. | Child ceiling and fork-controller body. |

Budget, stop conditions and the expected result shape are not new overlay
fields. Use the fork's `initial_invocation` or a later `ResumeRequest`:
`budget_limits`, `max_steps`, `limits` and `structured_output`. They are part
of the fork request identity when sent as `initial_invocation`.

## Enrolling a controller

```python
from cayu.fork_controllers import ForkController
from cayu.mcp import connect_mcp_toolset

compass_tools = await connect_mcp_toolset(
    McpServerSpec(
        name="compass",
        connection_id="compass",
        url="https://compass.example/mcp",
    )
)
app.enroll_fork_controller(
    ForkController(
        name="compass",
        subjects=("svc:compass",),
        agents=("finance-monthend",),
        host_tools=("lookup_ledger",),
        mcp_toolsets=(compass_tools,),
    )
)
```

Everything after `agents` is keyword-only.

- `subjects` are identities the host has authenticated. One subject acts for
  at most one controller.
- `agents` are the host agents the controller may fork. They must already be
  registered.
- `host_tools` are the only host tools any of the controller's forks may hold.
  The default is none. Each name must be a tool that at least one enrolled
  agent registers, so a misspelled name fails at enrollment. See
  [Host tool allowance](#host-tool-allowance).
- `mode` is the controller's declared operating mode. Only
  `ForkControllerMode.OBSERVE` (`"observe"`, the default) exists; any other
  value is rejected. See [Operating mode](#operating-mode).
- `mcp_toolsets` is optional. A controller toolset must use a URL transport,
  so its tools run on the controller's side, and may not use `secret_env` or
  `secret_headers`, so no host secret is resolved for it.

When a controller serves tools, enrollment registers one extra body per agent,
named `<agent>@fork-controller:<controller>`. It is the host agent's exact
registration (prompt, policies, hooks, context policy) plus the controller's
tools. The body runs only as an overlay fork from that controller: ordinary
runs, ordinary forks that name it, and host forks of its sessions are refused.

## Forking with an overlay

```python
controller = app.fork_controller_access("svc:compass")  # after host authentication
events = controller.fork(
    ForkSessionRequest(
        source_session_id=trunk_id,
        session_id="compass-accruals",
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
        execution_profile_selection="current_child",
        profile_adoption=ExecutionProfileAdoptionIntent(
            idempotency_key="compass-accruals",
            reason="Investigate the accrual variance.",
            requested_by=ResolutionActor(subject="svc:compass", source="request"),
        ),
        overlay=ForkOverlay(
            controller="compass",
            instructions="Investigate the September accrual variance only.",
            tools=(ForkOverlayToolGrant(tool_name="mcp__compass__lookup"),),
            memory=(ForkOverlayMemoryItem(id="aug", content="August closed at 2%."),),
        ),
    )
)
```

`controller.resume(ResumeRequest(...))` continues a fork this controller
created, while the host still enrolls the controller for that session's agent. `controller.snapshot_source(session_id)` returns the exact
`ForkSourceSnapshot` an `initial_invocation` fork needs; the child's first
invocation then runs through the host's ordinary durable dispatch.

`controller.enrollment()` returns a frozen `ForkControllerEnrollment`: the
controller name, the authenticated subject, `mode`, the enrolled `agents`,
`host_tools`, and `served_tools` (the names of the tools the controller
serves). It describes only the caller's own enrollment. Other controllers,
other subjects of the same controller and agents outside the enrollment never
appear.

Hosts that expose the entrance over HTTP mount
`cayu.server.fork_controller.create_fork_controller_router(app,
authenticate=...)`, which serves `GET /fork-controller/enrollment`,
`GET /fork-controller/sources/{id}/snapshot`, `POST /fork-controller/fork`
and `POST /fork-controller/resume`. The host's
`authenticate` callback names the subject; request bodies never do. A body's
`profile_adoption.requested_by` must name that same subject, and the recorded
actor is rebuilt from it with `source="http_auth"`; body `tenant` and `claims`
are dropped. A missing or empty subject returns 401. Denials, unknown sessions
and foreign sessions all return the same 403; a conflicting fork, a rejected
profile, or a source that changed after its exact snapshot returns 409.

### Bearer-token authentication

`ForkControllerBearerAuth` is a ready-made `authenticate` callback that maps
bearer tokens to subjects:

```python
from cayu.server.fork_controller import ForkControllerBearerAuth

# CAYU_FORK_CONTROLLER_TOKENS='{"<token>": "svc:compass"}'
authenticate = ForkControllerBearerAuth.from_environment()
# or ForkControllerBearerAuth.from_file("/run/secrets/fork-controller-tokens.json")
# or ForkControllerBearerAuth({"<token>": "svc:compass"})
app_server.include_router(create_fork_controller_router(app, authenticate=authenticate))
```

- The configuration is a JSON object from token to subject. Two tokens may
  name the same subject, so a token can be rotated without downtime.
- Each token must be an RFC 6750 bearer credential of at least 32 characters.
  At most 64 tokens are accepted.
- The callback keeps only SHA-256 digests of the tokens. It compares the
  presented token with every entry using `hmac.compare_digest`, without
  stopping at a match.
- A missing, repeated, malformed or unknown `Authorization` header
  authenticates no one, and the router returns 401.
- Configuration errors raise `AuthConfigurationError` and name only the entry's
  position. Tokens never appear in errors, logs or `repr`.

An overlay that grants tools changes the child's body, so it needs
`execution_profile_selection="current_child"` with a `profile_adoption` whose
`requested_by.subject` is the authenticated controller. An overlay with only
instructions and memory keeps the inherited profile and needs neither. Send it
with the default selection: built-in adoption covers only forks that grant
controller tools, so an instructions-only fork that asks for `current_child` is
rejected unless a configured `ExecutionProfilePolicy` adopts it.

## Authority rules

- `CayuApp.fork_session`, scoped resource access, runtime hooks and message
  text cannot attach an overlay. Only `ForkControllerAccess` can, and only for
  its own controller name and enrolled agents. The source session's agent must
  be enrolled, and the child keeps that agent: a fork cannot retarget a source
  outside the enrollment to an enrolled agent.
- An overlay fork belongs to the controller that created it, with or without
  granted tools: only that controller can fork it or snapshot it. Other
  controllers and ordinary host forks are refused, because the fork's durable
  overlay record names its owner.
- Agent names containing `@fork-controller:` are reserved for the bodies that
  enrollment registers.
- The host ceiling is still inherited or narrowed, and then limited to the
  controller's `host_tools` (see [Host tool allowance](#host-tool-allowance)).
  Controller tools never come
  from `tool_capability_ceiling`; only the overlay's grants add them, and each
  grant must name a tool the enrolled controller serves. The store accepts a
  child ceiling wider than its source only by tools the overlay grants, on a
  body named for that controller; re-granting a tool the source already had is
  not a widening.
- Controller tools are never inherited. A fork of an overlay fork keeps only
  the tools its own overlay grants.
- Host tool policy, approvals and runtime hooks apply to controller tools as to
  any tool. Tool policies see the host agent's name.
- Without an `ExecutionProfilePolicy`, Cayu adopts an overlay fork's profile
  only when the change is limited to the tool set (`direct_tools`,
  `tool_implementations`, `tool_view_grants`, `effect_authority`), the first
  invocation's budget, result shape and stop semantics
  (`invocation_budget_policy`, `structured_output`, `finalization`), and
  `runtime`, and only when the source's own registration is unchanged since it
  was created. If a host or controller registration changed in between, the
  change is not the overlay's, and adoption needs a configured policy. An
  unchanged profile, such as a re-fork of an overlay fork with the same tools,
  is adopted the same way. The decision is recorded with policy identity
  `cayu:fork-controller-overlay-adoption:v1`. With a configured policy, the
  policy decides.
- Agent-scoped budgets are counted by exact agent name, so they would not see a
  body's spend. While the host agent has an agent-scoped budget, runs of its
  fork-controller bodies fail closed. App and causal budgets apply as usual;
  forks share the source's causal budget.
- Scoped application subjects cannot resume or recover an overlay fork; the
  controller continues it through its own entrance.
- Tools in an overlay fork cannot publish to the host's knowledge store:
  `remember_knowledge` returns `publication_owner_closed`, so overlay memory
  never becomes host knowledge through an inherited tool.

## Host tool allowance

The host, not the controller, decides which host tools a controller's forks may
use. The controller's `tool_capability_ceiling` can only narrow that further.

- When a fork omits `tool_capability_ceiling`, the inherited host tools are
  narrowed to the enrollment's `host_tools`. With the default allowance, the
  fork has no host tools.
- When a fork's `tool_capability_ceiling`, its `initial_invocation`'s ceiling
  or a resume's ceiling names a host tool outside `host_tools`, the request is
  denied with `ForkControllerDenied` (403 over HTTP). This happens before any
  child is created and before any model or controller tool call. Cayu rejects the request instead of dropping the tool, so a
  controller never runs without a tool it asked for.
- Controller-served tools that the overlay grants are not part of the
  allowance and are unaffected by it.
- Provider-hosted tools such as OpenAI web search come from the agent's
  registration, not from the ceiling, so a fork cannot drop them. A controller
  can fork, replay or resume a fork of an agent registered with a hosted tool
  only when `host_tools` names that tool's type (for example
  `"openai_web_search"`). Otherwise the request is denied before any child or
  model request. Enrollment rejects a hosted type no enrolled agent registers.
- Every replay of an existing fork and every resume through the controller
  entrance checks the child's durable ceiling against the current allowance.
  If the host has since removed a tool from `host_tools` that the child's
  ceiling still names, the replay or resume is denied and the child's ceiling is not
  rewritten. Restoring the allowance makes the same fork replay again.

## Operating mode

`observe` is the only operating mode. An observing controller forks host
sessions, continues and reads its own forks, and describes its enrollment. The
controller entrance has no way to write to the session it forked from: it
cannot add messages to it, steer it, resume it or resolve its approvals, and a
fork cannot use it as the destination. Modes that write to the trunk are not
available.

## Instructions and memory

The overlay layer is one user-role message after the selected history:

```text
<cayu_fork_overlay version="1">
{"controller":"compass","instructions":"...","memory":[...],"scope":"this_fork_only"}
</cayu_fork_overlay>
```

It is rebuilt from the child's durable record for every model request. It is
not appended to the transcript, not written to any knowledge store, and does
not change the source session, its transcript or its registered body. Because
the source history and system prompt are unchanged, the shared prompt prefix is
preserved. The layer is added after context selection, so `instructions` and
`memory` together are capped at 32,000 characters
(`FORK_OVERLAY_CONTEXT_MAX_CHARS`). Overlay text containing a value from the
app's `SecretRedactor` is rejected; vault secrets that the redactor does not
know about are not checked.

## Identity, restart and results

The overlay is part of the fork request, so it is inside the fork request
digest that the child's lineage records. A generated child id is derived from
it. `session.forked` also carries `fork_overlay_controller`,
`fork_overlay_fingerprint` and `fork_overlay_tool_names`. The child stores the
exact overlay with its fingerprint under runtime-owned metadata.

- Retrying the same fork after a lost acknowledgement or restart returns the
  same child and the same `session.forked` event, while the controller is still
  enrolled for that child's agent; the source session need not still exist. A different overlay for the
  same `session_id` conflicts.
- A changed overlay record fails closed before any model request, and so does a
  fork-controller body session whose overlay record is missing.
- A changed controller tool contract changes the child's execution profile, so
  resume fails closed until the change is adopted. A missing enrollment makes
  the body unregistered and resume fails.
- MCP adapters with an explicit `connection_id` now have a portable
  execution-profile identity based on the server's tool contract, so controller
  tools keep their identity across processes.

Results come back as an ordinary child outcome. To deliver a finding to the
trunk, use the durable session-message path (`enqueue_session_message`), not
the overlay.
