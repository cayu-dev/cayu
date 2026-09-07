# Host-owned tool execution: verified boundary and durable delegation proposal

Status: characterization and design, not a new runtime API.
Baseline: `4b4adbfbe5e34251feafee6e911f532c81813a81`.

## What works today

A native `Tool.run(ctx, args)` can send a request to an application-owned
transport, await the host's result, and return `ToolResult`. The host performs
the effect; `CayuApp` continues to own model iteration, tool policy, pending
round checkpoints, tool-result publication, and the session transcript.

`tests/core/test_host_owned_tool_execution.py` verifies this with a scripted
provider and both in-memory and SQLite session stores. It also verifies that
policy denial prevents host execution and that validation in the native tool
rejects invalid input before delivery. The tests require no provider credentials
or external services.

This corrects an overly broad architectural assumption: host-owned execution
does **not** inherently require an application to implement its own model loop.
`ToolEffect.EXTERNAL` describes effects; it is not a transport switch.

Native tool implementations must validate arguments. Publishing an ordinary
native tool's JSON Schema to a provider is not equivalent to runtime validation
of every call against that schema.

## Live transport lifecycle

1. Start and continuously drain the `CayuApp.run(...)` event stream in an owned
   asynchronous task.
2. The authorized native tool sends its request to the host and awaits a result.
3. The host executes the action once and returns the corresponding result.
4. The native tool returns the result to Cayu. Cayu finalizes the tool round and
   advances the model using its normal transcript and runtime controls.
5. A subsequent customer message uses normal `CayuApp.resume(...)` on that
   session, after the previous invocation has completed.

A callback-style host may return control to its caller while the runtime task
continues waiting on the application's event loop. It must keep that loop and
task alive between callbacks. Tool results settle the waiting tool; they are
not customer messages and must not be injected through ordinary `resume`.

Applications need bounded queues, exact result correlation, deadlines, safe
argument/result handling, and explicit task cleanup. Cancelling a pending action
cannot undo an external effect that has already been dispatched. An invocation
idempotency key helps only if the execution owner honors it; it is not proof of
exactly-once execution.

## The narrower durable capability

A live coroutine cannot be reconstituted from a SQLite checkpoint. Existing
ordinary-tool recovery is for a crashed started-but-unresolved call, with
externally verified outcome evidence. It is not a normal pre-dispatch handoff
API. Deliberately creating a crash or fabricating a recovered result to implement
ordinary external execution would misrepresent authority and outcomes.

A first-class durable delegation feature would let the runtime checkpoint and
release its invocation **before** the host executes anything, then accept
authoritative results through an explicit resolution command. This supports
job queues, remote execution owners, and callback/webhook integrations as well
as benchmark environments. It is useful independently of any particular
domain, but it is not proven necessary for a live-process integration.

## Proposed contract; no public API shipped yet

The following names describe a proposal, not importable classes or methods:

- A schema-only delegated tool declaration with explicit effect classification
  and execution-profile identity.
- `PendingExternalToolRound`, carrying private execution authority and an
  explicitly authorized host projection. Public telemetry must not become
  execution authority or expose private arguments.
- A normal interruption reason such as `external_tool_results_required`.
- An explicit resolution command targeting the exact session instance,
  interaction, round, call, arguments/schema revision, and execution profile.

The command must reuse existing run fencing, atomic grouped transcript
publication, redaction, and durable publication receipts. It must not maintain
a parallel transcript or bypass ordinary authorization and budget enforcement.

Before implementation, settle these semantics:

| Boundary | Required decision and test |
| --- | --- |
| Admission | Schema validation, full-round policy, approval precedence, and exact executable argument authority before host delivery. |
| Mixed rounds | Which local calls may execute before or after handoff; no implicit repetition of already-completed siblings. |
| Partial results | Explicitly support durable partial settlement or reject incomplete batches atomically; never silently discard results. |
| Lost acknowledgement | An identical resolution replays its receipt; conflicting results fail without changing previously accepted outcomes. |
| External effect | Distinguish not dispatched, accepted, completed, failed, and unknown. Never equate cancellation with “no effect.” |
| Restart | Reconstruct the pending request without rerunning authorization or model generation and without executing a second mutation. |
| Concurrency | Reject stale workers, recreated sessions, conflicting resolution, and ordinary resume while the round is unresolved. |
| Context | Publish one complete assistant/tool group and preserve interaction, provider state, usage, and compaction authority. |
| Security | Host authorization, private/public projections, secret tracking, output limits, and scope must survive every transition. |

## Delivery gates

1. Keep this characterization independently green on unmodified runtime code.
2. Implement and verify a live application transport before attributing
   integration difficulty to a framework defect.
3. If process-release/restart delegation is required, implement its typed
   contracts and state transitions in separately testable logical commits.
4. Exercise interruption, approval, cancellation, duplicate/stale results, and
   publication failures against memory, SQLite, and PostgreSQL stores before
   treating the durable API as production-ready.

The characterization tests are not evidence that the proposed durable API,
cross-process result submission, or restart-safe delegation already exists.

## Runtime-first follow-up audit

`tests/core/test_runtime_reply_admission_contract.py` adds five credential-free
cases for the reply boundary. Ordinary `LoopPolicy.before_stop` can reject a
draft and continue through the runtime. Raw text events precede that gate, and
an interrupted `RunOutcome` can retain nonempty diagnostic `final_text`; a host
must require an accepted invocation before delivering a customer reply.
Structured-output validation intentionally takes a separate completion path
and does not run the generic before-stop gate. Do not assume schema-valid
structured output also passed an application reply policy.

Current runtime also has work contracts, deterministic completion verifiers,
bounded repeated-gap/attempt decisions, run limits and execution deadlines.
Workflow primitives govern application-defined stages; revisioned work context
and automatic recall support the runtime but do not authorize business effects.
An ordinary customer question or handoff may complete an interaction without
completing the customer's larger task.

The targeted audit at the baseline above passed 283 tests with one existing
failure and six PostgreSQL variants excluded. The failing resume-policy test
constructs a 16-step profile through `profiled_session_identity`, then resumes
using the current 64-step default. A diagnostic override aligning those limits
makes it pass. This is a test-maintenance finding; do not weaken profile fences.
Five separate existing before-stop tests also pass. No production runtime code
was changed by the audit.
