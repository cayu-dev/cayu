# Process-isolated host tools

Use `ProcessIsolatedTool` when trusted application code calls a synchronous
Python or native dependency that may not yield, honor cancellation, or release
the Python interpreter lock. It is an explicit opt-in adapter. Cayu does not
move ordinary `Tool` implementations into subprocesses and does not pickle live
tool objects.

## Choose the right boundary

These three claims are intentionally different:

| Claim | Meaning |
| --- | --- |
| `cooperative_in_process` | `CayuApp(tool_timeout_seconds=...)` requests cancellation, but the tool shares the worker interpreter. Code holding the GIL can prevent the deadline callback from running. |
| `hard_process_deadline` | `ProcessIsolatedTool` executes one invocation beneath a Linux subreaper supervisor. The supervisor owns a wall deadline and TERM-to-KILL cleanup of the complete adopted descendant tree without needing the tool interpreter to cooperate. |
| `sandboxed` | Filesystem, network, credential, privilege, or kernel isolation. The process adapter does not provide this claim and always publishes `sandboxed=false`. |

A runner, container, microVM, or remote service can provide a stronger security
boundary. Process isolation here is only a liveness and cleanup boundary for
trusted host-side code.

The selected boundary and timeout strength are part of the versioned
`direct_tools` execution-profile component. This includes ordinary tools: the
same declaration registered with and without application-wide
`tool_timeout_seconds` has intentionally different recovery authority. A
restart or recovery worker must use the matching timeout configuration; Cayu
rejects drift instead of silently weakening or strengthening an active
session's execution contract.

## Declare a reconstructable factory

Put the factory in an importable application module. It cannot be `__main__`, a
closure, a lambda, a live `Tool`, or an object captured from the parent.

```python
# my_app/isolated_handlers.py
from typing import Any

from cayu import ExecutionProfileBehaviorIdentity, ProcessIsolatedToolContext, ToolResult


class SearchHandler:
    def __init__(self, config: dict[str, Any]) -> None:
        self._endpoint = config["endpoint"]

    def run(
        self,
        context: ProcessIsolatedToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        # Call the trusted, possibly non-cooperative dependency here.
        return ToolResult(content=f"searched {arguments['query']}")


def build_search_handler(config: dict[str, Any]) -> SearchHandler:
    return SearchHandler(config)


build_search_handler.execution_profile_identity = ExecutionProfileBehaviorIdentity(
    name="my-app:isolated-search-factory",
    behavior_version="1",
    implementation_version="2026-08-25",
)
```

Register the adapter as an ordinary agent tool:

```python
from cayu import (
    ExecutionProfileBehaviorIdentity,
    ProcessIsolatedTool,
    ProcessIsolatedToolContextProjection,
    ProcessIsolatedToolFactoryRef,
    ProcessIsolatedToolLimits,
    ToolEffect,
    ToolSpec,
)

search = ProcessIsolatedTool(
    ToolSpec(
        name="isolated_search",
        description="Run the application search adapter.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        effect=ToolEffect.NONE,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="my-app:isolated-search",
            behavior_version="1",
            implementation_version="2026-08-25",
        ),
    ),
    factory=ProcessIsolatedToolFactoryRef(
        module="my_app.isolated_handlers",
        qualname="build_search_handler",
        identity=ExecutionProfileBehaviorIdentity(
            name="my-app:isolated-search-factory",
            behavior_version="1",
            implementation_version="2026-08-25",
        ),
    ),
    limits=ProcessIsolatedToolLimits(
        deadline_seconds=10,
        term_grace_seconds=1,
        kill_grace_seconds=1,
    ),
    factory_config={"endpoint": "https://search.internal.example"},
    context_projection=ProcessIsolatedToolContextProjection(fields=("session_id",)),
    environment={"SEARCH_REGION": "us-east-1"},
)
```

The module must be importable by the same Python installation under isolated
interpreter mode (`python -I`). The imported callable must expose an
`execution_profile_identity` equal to the identity declared by
`ProcessIsolatedToolFactoryRef`; advance both copies together when the factory
behavior changes. A missing, changed, or identity-mismatched factory fails as a
bounded typed child error. Cayu never falls back to importing a caller-supplied
object or running it in process.

## Parent and child authority

The trusted parent completes lookup, JSON Schema validation, policy and
approval, execution-profile selection, effect classification, and runtime
idempotency construction before creating the child. It then sends one
canonical, versioned, size- and node-bounded JSON document over stdin. The
document binds the factory, limits, validated arguments, projected context,
declared environment, session, and complete runtime-owned tool-call identity.
After those checks and before process creation, the parent atomically records
an exact isolated-dispatch preparation/possible-admission record plus a
separate immutable authority record. If process-boundary setup, supervisor
spawn, caller cancellation, the pre-admission deadline, or the final
process-cleanup fence then settles with positive proof that the one-shot
worker-admission signal did not cross, the parent records an exact
`worker_not_admitted` settlement bound to that preparation. Recovery
reconstructs the stable call and
environment authority from the pending round, authenticates the authority
record that owns the request/effective-argument digests, and requires the
preparation record to match both. An authenticated zero-dispatch settlement
overrides the earlier conservative preparation; neither representation relies
on the generic tool-started event.

The child receives only:

- the bounded `factory_config` JSON object;
- the validated arguments;
- fields and metadata keys named by `ProcessIsolatedToolContextProjection`;
- the explicitly declared string environment, plus fixed locale/UTF-8 values;
- stdin, bounded stdout/stderr pipes, and one private result descriptor.

It does not inherit stores, runners, workspaces, event loops, provider clients,
policy or approval objects, open parent sockets, `PATH`, `HOME`, proxy settings,
cloud/model credentials, or any other parent environment. An application may
explicitly declare a non-interpreter environment value when its adapter needs
it. `PYTHON*`, dynamic-loader, and interpreter-affecting environment names are
always rejected. The current adapter also rejects registered workload-secret
values in configuration, arguments, projected context, or environment. A tool
needing credentials must use a separate application-owned broker/acquisition
design; ambient inheritance is not a compatibility path.

The worker returns exactly one length-framed canonical terminal envelope
containing a valid `ToolResult` or a fixed error code. The frame lets the parent
begin owned supervisor settlement without waiting for every descendant-held
descriptor to close, while the retained reader continues rejecting trailing or
multiple output through settlement. Invalid UTF-8/JSON, non-canonical or
trailing data, multiple frames, identity disagreement, missing output, output
overflow, child exceptions, signals, and crashes become bounded runtime-owned failures.
Child diagnostics and stdout/stderr are never copied into model-facing errors.
The successful result still goes through Cayu's ordinary result validation,
redaction, artifact projection, transcript, hook, and event boundaries.

## Deadline, effects, cancellation, and recovery

The wall deadline starts before process creation. On success, failure, timeout,
or caller cancellation, the parent closes an invocation-specific inherited
control channel to ask the Linux subreaper supervisor to close the invocation.
The supervisor cannot create its worker until the parent has received the exact
supervisor process handle, installed its wait/settlement owner, and sent a
one-shot admission signal. A parent-side subprocess transport failure, timeout,
or cancellation before that handoff closes the channel and remains zero-dispatch.
The parent never signals the supervisor through a reusable numeric PID or
process-group identity. The supervisor signals the worker group and every adopted
descendant while the worker group remains owned, stops using the numeric group
identity after its leader is reaped, waits for the declared TERM grace,
escalates to KILL, reaps the complete tree, and only then exits. Reaping retains
only constant-size worker-leader status; completed descendant history is not
accumulated for the lifetime of the invocation. Registration
first runs a supervisor capability probe that must actually enable Linux
child-subreaper ownership and read the kernel child-ownership interface;
successful proof is cached by process generation, while failed probes remain
retryable. Filesystem shape alone is not support evidence. The parent requires
both the exact invocation-supervisor wait and that supervisor's post-reaping
acknowledgement channel before accepting cleanup proof. The acknowledgement
proves complete tree reaping and carries a typed supervisor-health outcome
separate from the worker status propagated by the supervisor process. A
`supervisor_failed` outcome therefore produces a bounded failure even if a valid
terminal frame exists, remains observable after delayed spawn settlement, and
retains independent terminal-protocol or diagnostic failures as ordered causal
evidence. A healthy supervisor does not turn an independently classified worker
failure into a supervisor failure. A cancelled or failed wait
or a missing/malformed acknowledgement is not cleanup proof; the retained
lifecycle owner and dispatch fence remain unresolved.
Caller `Task.cancel()` remains normal `CancelledError`, with its task cancellation
requests restored, after that settlement.
While any retained isolated-process cleanup owner remains unresolved, the
process rejects later isolated child dispatches before spawn. Settled cleanup
removes that fence; ordinary in-process tools are unaffected.

Execution-boundary controls in transcript and provider projections retain their
runtime-authored values even when those values collide with a registered secret.
Application-authored tool results cannot claim that exemption by reproducing the
same dictionary shape. Before-hook synthetic results and after-hook replacements
are copied, stripped, and redacted before durable hook-completion publication.
Positively recognized runtime-control fields are also stripped at the shared
terminal-publication boundary for direct and recovered results, and again at
untrusted message ingress before any later runtime projection. When the runtime
restores its exact control tuple after an untrusted hook replacement, it first
removes every caller-shaped reserved sibling so a malformed partial tuple
cannot poison the restored authority.

Killing local processes does not prove that an external effect did not happen.
The tool's `ToolEffect` remains authoritative. Timeout or abnormal termination
of `IDEMPOTENT` or `EXTERNAL` tools publishes `outcome_unknown=true`; `EXTERNAL`
also requires manual reconciliation. Cayu does not replay the invocation merely
because its child was killed. Duplicate delivery and recovery continue to use
the ordinary runtime-owned tool-execution identity and receipt rules.
An exact `worker_not_admitted` settlement overrides the earlier conservative
preparation marker for both automatic and manual recovery; an operator cannot
assign an executed outcome to a call the runtime positively knows never crossed
worker admission.

The first implementation requires Linux `PR_SET_CHILD_SUBREAPER` and `/proc`
child enumeration. Unsupported
platforms reject registration instead of advertising a hard deadline. Abrupt
death of the Cayu parent is a separate deployment/process-supervision concern;
this adapter does not claim to solve it.

Inspect `CayuApp.describe()` or registered capability evidence to verify
`execution_boundary`, `timeout_strength`, adapter identity/configuration digest,
deadline, and `sandboxed=false` without exposing factory configuration or
environment values. When the application-wide `tool_timeout_seconds` is shorter
than the adapter deadline, the published hard deadline is the shorter effective
value because cancellation of this adapter still enters owned process cleanup.
