# Durable service-backed tools

For a complete installed-package support agent with restartable clarification and
external approval receipts, run `cayu guide order-support`.

Use this recipe when a custom tool wraps an application-owned service and must
resume a saved session in a replacement process. For the complete proposal,
approval, action, verification, and recovery lifecycle, keep using
`cayu guide durable-operations`. This guide supplies the component identity and
knowledge bindings needed when reconstructing that application.

## Two separate bindings

| Configuration | What it establishes | What it does not establish |
| --- | --- | --- |
| `SQLiteSessionStore` or another persistent session store | Durable conversation, checkpoints, and pending actions | Equivalent tool, policy, or environment behavior after rebuilding the app |
| `ExecutionProfileBehaviorIdentity` | An explicit versioned declaration of component behavior | Persistence, authorization, or downstream idempotency |
| `CayuApp(knowledge_store=store)` | Application-level knowledge configuration | An ordinary tool's `ctx.knowledge_store` |
| Injected bound store, or selected environment's store and scope | The knowledge service the tool can actually query | Authority outside the supplied scope |

A direct `tool.run()` test misses runtime registration and admission. Test a
real `CayuApp`, persist a session, and reconstruct the app in a second OS
process. An undeclared opaque service object may make the reconstructed
execution profile incompatible even when its Python class and tool name match.

## Declare component identity

```python
from cayu import ExecutionProfileBehaviorIdentity

reader_identity = ExecutionProfileBehaviorIdentity(
    name="handbook-reader",
    behavior_version="1",
    implementation_version="1",
)
```

Place this declaration on `ToolSpec.execution_profile_identity`. A custom
policy returns its own declaration from the
`ToolPolicy.execution_profile_identity` property. A registered custom
environment declares its own `EnvironmentSpec.execution_profile_identity`.
Declare each behavior-bearing component, not just the tool:

```python
from cayu import EnvironmentSpec, ToolPolicy

class HandbookPolicy(ToolPolicy):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="handbook-policy", behavior_version="1", implementation_version="1"
        )

    # Implement authorize(request) for your application's actual authority rules.

environment_spec = EnvironmentSpec(
    name="handbook",
    execution_profile_identity=ExecutionProfileBehaviorIdentity(
        name="handbook-environment", behavior_version="1", implementation_version="1"
    ),
)
```

Keep the logical name stable. Change `behavior_version` when externally
observable semantics or relevant non-secret configuration changes; change
`implementation_version` for every implementation deployment, even when its
public contract remains the same. Encode configuration revisions in these
versions; the identity has no arbitrary configuration field. Do not put secrets,
credentials, object addresses, random UUIDs, process IDs, or request IDs here.
Ordinary service data can evolve under an unchanged retrieval contract.

This declaration asserts equivalent behavior across reconstruction. It does not
prove equivalence or make incompatible changes safe. On a legitimate profile
change, start a new session or follow the explicit execution-profile adoption
contract in `cayu guide anatomy`; never automatically adopt on mismatch, disable
checks, or reuse an old version to conceal changed behavior. Operation
idempotency keys and durable approval/round/call IDs serve different purposes;
see `cayu guide tool-effects` and `cayu guide durable-operations`.

## Inject the supplied knowledge store

Constructor injection is the smallest ordinary-tool pattern. Validate a
required dependency before registering the tool or issuing a provider request:

```python
import json
from cayu import KnowledgeQuery, KnowledgeStore, Tool, ToolEffect, ToolResult, ToolSpec

class HandbookReader(Tool):
    spec = ToolSpec(
        name="lookup_handbook",
        description="Search the authorized handbook.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.NONE,
        execution_profile_identity=reader_identity,
    )

    def __init__(self, store: KnowledgeStore):
        if not isinstance(store, KnowledgeStore) or store.bound_access_scope() is None:
            raise ValueError("HandbookReader requires the supplied, bound knowledge store")
        super().__init__()
        self.store = store

    async def run(self, ctx, args):
        found = await self.store.search(KnowledgeQuery(text="shipping", namespace="handbook"))
        payload = found.model_dump(mode="json")
        return ToolResult(content=json.dumps(payload), structured=payload)

```

Register `HandbookReader(the_supplied_scoped_store)` on your agent.

Use the actual supplied service. Do not construct an empty replacement store or
silently return empty evidence when a required resource is missing. A namespace
filter narrows retrieval; it is not authorization. The hosting application
derives the bound scope from trusted principal/session ownership. Keep tenant
or organization restrictions, labels, visibility, and other scope constraints;
never accept model-selected tenant identity or broaden the scope to fix a miss.
Inject a store bound to that authority at the appropriate application lifetime.

## Bind through an environment

When tools should consume environment resources, explicitly bind and select the
environment instead. Given the same supplied bound `store`:

```python
from cayu import Environment

environment = Environment(
    environment_spec,
    knowledge_store=store,
    knowledge_access_scope=store.bound_access_scope(),
)
```

Call `app.register_environment(environment)` and select
`environment_name="handbook"` in `RunRequest`.

Inside the tool, require `ctx.knowledge_store` and `ctx.knowledge_access_scope`,
then call `ctx.knowledge_store.search(query, access_scope=ctx.knowledge_access_scope)`.
An explicit scope must match the store's bound scope. Application-level
knowledge configuration alone does not populate these context fields; without
a selected environment binding they may be `None`. Built-in knowledge tools
use this environment path too; see `cayu guide references#knowledge`.

## Prove reconstruction

The repository example `examples/durable_service_tools/app.py` is credential-free
and uses public APIs with SQLite sessions and a scripted provider. Run `start`
and `resume` against one fresh directory in separate OS processes. It verifies
that the supplied store is searched, current scoped evidence reaches the
provider, forbidden records stay excluded, and the earlier conversation survives.
Use `--wiring environment` on both commands to exercise the environment path.

The example also supplies a versioned custom policy and a small SQLite receipt
tool. Its `pause`, `approve` or `deny`, and `repeat` phases exercise durable
approval reconstruction and completed-resolution behavior. This local fixture
proves only its own atomic insert/idempotency contract, not generic exactly-once
delivery to external services. Product handlers still authenticate and authorize
the resolver against exact review evidence as shown in `cayu guide durable-operations`.

Negative checks matter: a changed tool, policy, or environment version must fail
before a protected effect or provider request; an opaque undeclared tool must
remain incompatible across reconstruction. For a real service, also test
unknown external outcomes and reconciliation using `cayu guide tool-effects`.

## Reading admission errors

`app.run` creates a new session. For an ordinary next conversational turn:

```python
from cayu import CayuApp, Message, ResumeRequest

async def next_turn(app: CayuApp, existing_id: str) -> None:
    async for event in app.resume(ResumeRequest(
        session_id=existing_id,
        messages=[Message.text("user", "Next turn")],
    )):
        print(event.type)
```

An existing session may instead be running, awaiting approval/input, or require
recovery. Inspect its status and pending actions before choosing the corresponding
resolution/recovery API (`cayu guide references#sessions`). Ordinary resume does not resolve
an approval.

`ExecutionProfileMismatchError.differences` reports bounded class-level categories:
`opaque_identity` means at least one compared class has process-local identity;
`other_or_unknown` means the available evidence cannot diagnose the change.
An opaque category is evidence about identity strength, not proof that reconstruction
caused the mismatch. Declare stable behavior and implementation identities from the
first run. Adding one later does not repair an existing opaque baseline.

The persisted profile format retains aggregate digests and identity strength, not
individual member identifiers or declared versions. No durable evidence extension is
introduced here. Consequently errors cannot distinguish a changed declared version
from a member addition/removal, nor name the member responsibly. Inspect the persisted
execution-profile decision, its changed classes, and the application's declarations.
For real behavior changes start a new session or follow explicit profile adoption;
never reuse an old version to conceal a change. Adoption rejection and migration
requirements remain distinct outcomes.
