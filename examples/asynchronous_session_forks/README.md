# Asynchronous session forks

This API-key-free tracer shows the Runtime primitive beneath councils,
tournaments, and dynamic worker trees: exact session forks that enter ordinary
durable dispatch and settle independently.

The application completes trunk checkpoint 14, captures one
`ForkSourceSnapshot`, and creates caller-named children A, B, C, and D from that
same asserted authority. Each fork binds its exact first `ResumeRequest` and
sole `initial_dispatch_id`; each matching `DispatchRequest` returns while every
child model call is still zero. An alternate dispatch identity or inline resume
cannot consume the frozen first invocation.
The trunk then advances to checkpoint 15.

Deterministic `asyncio.Event` barriers—not timing sleeps—control the workers:

- B completes first and its transcript is consumed by another ordinary session
  while A, C, and D remain running;
- D fails without changing any sibling;
- A later performs one idempotent tool effect and completes;
- C is interrupted without cancelling A or B; and
- an ordinary evaluator session reads the settled A/B outputs, after which the
  application chooses the named result.

There is deliberately no Runtime join object. Parent lineage, the exact source
snapshot, causal-budget identity, queue task identity, and bounded application
metadata provide correlation. The example owns result consumption and
selection.

The tracer also injects queue-publication and task-terminal acknowledgement
loss, cancels one worker after its durable claim, and simulates hard worker loss
immediately after both durable provider completion and durable session-terminal
publication. Reconstructed producer and worker applications retry every fork
and dispatch. The tracer asserts that child identity, queue identity, provider
completions, the tool mutation, and terminal results are not duplicated.

Run it with:

```bash
uv run python -m examples.asynchronous_session_forks.app
```

The final JSON is bounded evidence intended for inspection. A nonzero exit
means one of the runtime invariants failed.

For a slower durability proof that does not share stores, event loops, or
Python objects, run:

```bash
uv run python -m examples.asynchronous_session_forks.process_recovery
```

This second tracer creates each exact fork and queue task in a setup process,
then terminates a worker immediately after its SQLite-backed claim, durable
provider completion, or durable terminal-event commit. A fresh process advances
the task-store clock beyond the abandoned lease, reclaims it through the public
store contract, replays producer admission, and reconciles the child to a
terminal result. A completion without a recoverable provider-operation handle
fails closed as interrupted; the committed terminal-publication case remains
completed. The tracer never calls `release_task`; the durable provider-call log
must contain exactly one source call and one child call in every scenario.
