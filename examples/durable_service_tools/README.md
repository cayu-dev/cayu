# Durable service-backed tools

The canonical explanation is package-shipped: `cayu guide durable-service-tools`
([source](../../src/cayu/guides/durable-service-tools.md)). The existing
`cayu guide durable-operations` remains the complete operations lifecycle recipe.

From a Cayu checkout with its existing development environment, use a fresh
state directory and separate commands (each starts a new OS process):

```sh
state_dir=$(mktemp -d)
uv run python examples/durable_service_tools/app.py start "$state_dir"
uv run python examples/durable_service_tools/app.py resume "$state_dir"
# Intentional ExecutionProfileMismatchError, before provider/tool execution:
uv run python examples/durable_service_tools/app.py resume "$state_dir" --version 2
```

Repeat with a new directory and `--wiring environment` on both successful
commands. `--policy-version 2` and `--environment-version 2` test the corresponding
incompatible registrations (the latter requires environment wiring).
`--opaque` on both start/resume demonstrates fail-closed undeclared tool identity.

For the bounded approval fixture:

```sh
approval_state=$(mktemp -d)
uv run python examples/durable_service_tools/app.py pause "$approval_state" --wiring environment
uv run python examples/durable_service_tools/app.py approve "$approval_state" --wiring environment
uv run python examples/durable_service_tools/app.py repeat "$approval_state" --wiring environment
```

Use `deny` instead of `approve` with another fresh directory to verify no receipt
is written. The trusted local demo resolver is not a product authentication
handler. SQLite receipt uniqueness is specific to this fixed-action fixture;
it is not a guarantee for arbitrary external effects.

No credentials, network, live model, or external database are needed. The
knowledge service is rebuilt with current evidence each process; assertions
inspect actual supplied-store calls and provider-visible history. Session
connections are closed before exit. Tests inspect persisted observations and
receipts independently, including failure paths:

```sh
uv run pytest tests/examples/test_durable_service_tools.py tests/cli/test_guide.py
```
