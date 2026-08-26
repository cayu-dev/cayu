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
| `hard_process_deadline` | `ProcessIsolatedTool` executes one invocation in a new POSIX process session. The parent owns a wall deadline and bounded TERM-to-KILL process-group cleanup without needing the child's interpreter to cooperate. |
| `sandboxed` | Filesystem, network, credential, privilege, or kernel isolation. The process adapter does not provide this claim and always publishes `sandboxed=false`. |

A runner, container, microVM, or remote service can provide a stronger security
boundary. Process isolation here is only a liveness and cleanup boundary for
trusted host-side code.

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
After those checks and immediately before process creation, the parent durably
records the exact isolated-dispatch admission. Recovery treats that marker—not
the earlier generic tool-started event—as positive evidence that child
execution may have been admitted.

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

The child returns exactly one length-framed canonical terminal envelope
containing a valid `ToolResult` or a fixed error code. The frame lets the parent
begin owned process-group settlement without waiting for every descendant-held
descriptor to close, while the retained reader continues rejecting trailing or
multiple output through settlement. Invalid UTF-8/JSON, non-canonical or
trailing data, multiple frames, identity disagreement, missing output, output
overflow, child exceptions, signals, and crashes become bounded runtime-owned failures.
Child diagnostics and stdout/stderr are never copied into model-facing errors.
The successful result still goes through Cayu's ordinary result validation,
redaction, artifact projection, transcript, hook, and event boundaries.

## Deadline, effects, cancellation, and recovery

The wall deadline starts before process creation. On success, failure, timeout,
or caller cancellation, the parent closes the invocation, signals the complete
owned process group, waits for the declared TERM grace, escalates to KILL, and
requires both the direct child and group to disappear before cleanup is proven.
Caller `Task.cancel()` remains normal `CancelledError` after that settlement.
While any retained isolated-process cleanup owner remains unresolved, the
process rejects later isolated child dispatches before spawn. Settled cleanup
removes that fence; ordinary in-process tools are unaffected.

Killing local processes does not prove that an external effect did not happen.
The tool's `ToolEffect` remains authoritative. Timeout or abnormal termination
of `IDEMPOTENT` or `EXTERNAL` tools publishes `outcome_unknown=true`; `EXTERNAL`
also requires manual reconciliation. Cayu does not replay the invocation merely
because its child was killed. Duplicate delivery and recovery continue to use
the ordinary runtime-owned tool-execution identity and receipt rules.

The first implementation requires POSIX process-group support. Unsupported
platforms reject registration instead of advertising a hard deadline. Abrupt
death of the Cayu parent is a separate deployment/process-supervision concern;
this adapter does not claim to solve it.

Inspect `CayuApp.describe()` or registered capability evidence to verify
`execution_boundary`, `timeout_strength`, adapter identity/configuration digest,
deadline, and `sandboxed=false` without exposing factory configuration or
environment values. When the application-wide `tool_timeout_seconds` is shorter
than the adapter deadline, the published hard deadline is the shorter effective
value because cancellation of this adapter still enters owned process cleanup.
