# Lineage-scoped shared artifacts

Use Cayu's shared-artifact tools when a parent agent creates a file that an
isolated fork or subagent must receive explicitly. Keep transient exploration in
the parent workspace; publish only a validated handoff file.

## Application setup

Register the same immutable policy on every agent that may publish or
materialize the handoff. The environment must expose a durable `ArtifactStore`;
production restart recovery also requires a durable `SessionStore`.

```python
from cayu import (
    MaterializeSharedArtifactTool,
    PublishWorkspaceArtifactTool,
    SharedArtifactPolicy,
)

policy = SharedArtifactPolicy(
    publish_path_prefixes=("handoff",),
    materialize_path_prefixes=("received",),
    allowed_content_types=("text/plain", "text/x-python"),
    max_bytes=4 * 1024 * 1024,
    max_publications_per_session=32,
    grant_ttl_seconds=6 * 60 * 60,
    max_lineage_depth=16,
    retention_class="lineage_handoff",
    allow_overwrite=False,
)

tools = (
    PublishWorkspaceArtifactTool(policy),
    MaterializeSharedArtifactTool(policy),
)

# app.register_agent(parent_spec, tools=(*parent_tools, *tools))
# app.register_agent(child_spec, tools=(*child_tools, *tools))
```

The policy is application authority. Do not let model input choose path prefixes,
content types, byte limits, audience, retention class, lineage depth, expiry, or
overwrite behavior.

## Explicit model-facing sequence

The parent performs:

```json
{"path":"handoff/solver.py"}
```

with `publish_workspace_artifact`. Its successful tool result content is one
canonical string beginning with `cayu-shared-artifact-v1.`. Put that complete
string in the child task or fork message; do not reduce it to an artifact id.

The child performs:

```json
{
  "ref":"cayu-shared-artifact-v1.<complete returned token>",
  "destination":"received/solver.py"
}
```

with `materialize_shared_artifact`, then uses ordinary governed tools such as
`read_file` or `exec_command` on `received/solver.py`. The child cannot infer the
reference from `list_artifacts`, and an unrelated session cannot materialize a
leaked reference.

Handoff does not grant a vault, proxy, or credential handle. Cayu refuses
publication when the source file contains an exact secret registered in the
parent's current invocation, repeats that check against the child's current
invocation before materialization, and never writes a redacted substitute for
the exact file. This is not taint tracking or general DLP: keep the policy's
allowed paths credential-free and govern every tool that can write them.

## Fresh-process recovery check

Run this acceptance sequence before relying on handoff in production:

1. Configure `SQLiteSessionStore` or `PostgresSessionStore` and a durable local
   or remote artifact store with an explicit stable `store_id`.
2. Start a parent session, create the child through Cayu's fork or subagent
   boundary, publish a file, and durably record the returned opaque ref in the
   child request or application state.
3. Stop the parent application process completely.
4. Construct a new `CayuApp` process against the same session database and the
   same artifact bytes and `store_id`.
5. Resume the already-created child, materialize the ref into its newly bound
   workspace, and execute or inspect the reconstructed file.
6. Repeat with the ref in an unrelated same-environment session and require the
   fixed `authorization_denied` result.

A copied workspace, reused process object, matching directory name, or newly
named artifact store is not restart evidence. The durable grant, exact lineage,
stable store identity, content digest, metadata, and receipt must all verify.

## Lifecycle ownership

Publishing means “available for this bounded descendant handoff,” not “part of
the agent forever.” The application or evolving agent still classifies the file:

```text
scratch   -> leave in the mutable workspace and expire it
evidence  -> retain the artifact and reference it from the run/build receipt
promoted  -> validate it and capture it through the AgentSnapshot component policy
```

`materialize_shared_artifact` never auto-promotes a file into agent anatomy.
Revocation uses `revoke_shared_artifact_grant(session_store, ref, reason=...)` and
preserves the revoked record as evidence.
