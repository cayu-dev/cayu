# Durable human-attention notifications

Read the canonical package guide: `cayu guide human-attention`
([source](../../src/cayu/guides/human-attention.md)).

This provider-neutral fixture uses a persistent Runtime store, an at-least-once
EventSink, a separate durable destination, restart reconciliation, and exact-action
resolution. No credentials, live model, or network are needed for the CLI proof.

```sh
attention_state=$(mktemp -d)
uv run python -m examples.human_attention.app pause "$attention_state"
uv run python -m examples.human_attention.app consume "$attention_state"
uv run python -m examples.human_attention.app inspect "$attention_state"
uv run python -m examples.human_attention.app answer "$attention_state"
uv run python -m examples.human_attention.app consume "$attention_state"
uv run python -m examples.human_attention.app late-hint "$attention_state"
```

Use a fresh directory for each scenario. For approvals, add `--kind tool_approval`
to `pause`, then use `approve` or `deny`. `interrupt` supersedes a question through
Runtime. The destination's `inspect` output contains no question or tool arguments.
The local demo operator is trusted; production recipient/session authorization
belongs in the consuming application.

Fault flags: `pause --crash-before-delivery` exits 17 after the committed pause;
`pause --fail-after-accept` loses a hint acknowledgement; `consume
--crash-after-accept` exits 18 after a durable notification insert. Resume each
case with a fresh `consume` process. Concurrent consumers deduplicate in SQLite.

The consumer enrolls all current actions in its configured scope, including pauses
that predate enrollment. It has bounded pages and does not infer closure from
absence or a failed query. The same consumer supports manual-recovery projections;
the CLI producer fixtures demonstrate questions and approvals. Manual outcomes
must be independently verified and supplied through Runtime's recovery APIs.

To use the optional authenticated server with the same state directory:

```sh
export ATTENTION_STATE="$attention_state"
export ATTENTION_SERVER_USERNAME=operator
# Set ATTENTION_SERVER_PASSWORD through your deployment's secret configuration.
uv run uvicorn examples.human_attention.server:create_demo_server --factory
```

The server owns periodic event recovery. Run `consume` independently on your
scheduler. Use protected `/api/pending-actions` for public action references and
the normal protected resolution routes. Never expose the demo SQLite destination
directly as an unauthenticated inbox.

```sh
uv run pytest tests/core/test_human_attention.py tests/examples/test_human_attention_example.py
```
