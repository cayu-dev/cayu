# Tools, knowledge and use reported to Cloud

When its server starts, a Runtime that Cayu Cloud deployed tells Cloud which tools
and knowledge the running Release has, then reports how often its specialists use
them. This is the Runtime side of Cloud's **`cayu.agent-registry/v1`** contract
(`docs/contracts/agent-registry-v1.md` in the Cloud repository). Cloud shows what
the Runtime sends on the Agent's Tools and knowledge tab.

`/cloud/contract` also lists `tool_knowledge_registration` and
`tool_knowledge_usage` in `features`; an older Runtime that lacks them never
registers.

## Enrollment and lifetime

Cloud binds these variables on the Agent's web process:

| Variable | Meaning |
| --- | --- |
| `CAYU_CLOUD_API_URL` | HTTPS Cloud API base URL |
| `CAYU_CLOUD_APPLICATION_ID` | The Agent's id in Cloud |
| `CAYU_CLOUD_REGISTRY_TOKEN` | The Agent's own registry credential (`cayu_reg_…`) |
| `CAYU_RELEASE_ID` | The running Release; `CayuApp(release_identity=...)` also works |

Without the registry token, nothing is started. With it, a missing or invalid URL,
Agent id or Release fails server startup. The token works only on this Agent's
registry write routes and reads nothing; it is separate from the sandbox token,
the Cloud reader's token and the Developer dashboard login.

`create_server`, `mount_cayu` and `cayu serve` own the reporter's lifetime. A
custom host can enter `cayu.cloud_inventory.cloud_inventory_lifespan(app)` after
registering its specialists and before serving work, and exit it before closing
the app's stores. Pass `project_root` so project tools are classified correctly;
`cayu serve` passes its resolved project root. Environment factories are not
materialized for discovery.

The reporter registers at once, then reports use every 30 seconds, draining a
backlog without waiting. Network and server failures log one content-free warning
and retry after 30 seconds without stopping agent work. When Cloud refuses the
credential, the Agent or the registration itself (401, 404, 413 or 422), the
reporter logs a warning and stops until the service restarts. Shutdown cancels it
before the app closes. Requests have a 10-second timeout, ignore proxy variables
and never follow redirects.

## Register

`PUT {CAYU_CLOUD_API_URL}/v1/applications/{application_id}/agent-registry/releases/{release_id}`

The body (`cayu.cloud_inventory.CloudRegistration`) names the Runtime version,
`store_id` (the session store's identity from `SessionStore.read_store_identity()`,
left out when the store has none, such as a database still at storage revision
124), `start_cursor` (the newest event sequence in the session store when the
process started, or, when it registers again, where its own reports reached), the
specialists, the tools and the knowledge entries. Cloud keeps a usage cursor for
each session store of the Agent and answers with the registering store's, which the
reporter then follows. The same store registering again keeps its cursor; a store
Cloud has no cursor for (an in-memory store after a restart, a replaced database)
starts at its `start_cursor`. If the store's identity changes under a running
process, the reporter registers the new store, counting from its newest event.

Each tool is reported once by name, with every specialist that has it, its
declared description, effect, approval and source:

- `approval` comes from the manifest's static policy coverage: `never`
  (allowed), `always` (approval required), `sometimes` (the policy decides per
  call), `blocked` (denied) or `unknown`.
- `source.kind` is `agent` for the project's own tools, `cayu` for built-in
  tools, `package` with the installed distribution and version for a tool from
  another package, `mcp` with the configured server name (not its URL, command,
  environment, credentials or instructions), `provider` for a model provider's
  hosted tool, or `unknown`.
- If specialists disagree about a same-name tool, approval and source become
  `unknown` when those fields differ, the most consequential declared effect is
  retained, and differing descriptions are omitted.
- An MCP tool's own description is sent, without the server instructions Runtime
  adds for the model. Argument schemas are not sent.

Knowledge is limited to what application code registered. An entry is reported
only when it is active and unexpired, was created by the application
(`created_by_type=KnowledgeActorType.APP`), has shared visibility (global,
organization, project or workspace) and was not written by Cayu's curator or
knowledge maintenance. Model-written memories (whether pending or kept), entries
people created, per-user, per-session and per-task entries, and pending, archived,
deleted or expired entries never leave the Runtime. For each reported entry Cloud
receives its id, title, kind and update time, never its text, previews, labels,
sources, metadata or author. Registration reads the app's knowledge scope and
statically registered environments' scopes; it does not elevate them.

Cloud refuses values over its limits instead of truncating them, so Runtime fits
the body first: at most 100 specialists, 500 tools and 500 entries (`truncated`
is set when more entries qualify), descriptions cut to 2,000 characters and titles
to 300, and a body under 1 MiB. It drops entries before shortening descriptions,
then keeps only the largest prefix of tools that fits if specialist associations
still exceed the byte limit. Usage is counted only for the transmitted tools.
Workload secrets and the registry token are redacted before normalization or
truncation. Tool names and entry ids containing secrets are omitted rather than used
as usage identifiers.

## Report use

`POST {CAYU_CLOUD_API_URL}/v1/applications/{application_id}/agent-registry/releases/{release_id}/usage`

A report (`CloudUsageReport`) counts the events with a sequence in
`(after, through]`, where `after` is Cloud's cursor. It holds totals, never
events: `calls` (completed or failed tool calls), `failures`, `approved` and
`rejected` (people's approval decisions) and the last use per tool, and `lookups`
and the last use per knowledge entry. Use is counted only for the tools and
entries this Release registered, so the body is bounded by the registration and
the ids of unregistered entries never leave the Runtime. Provider-hosted web
searches count terminal calls, with failed calls also counted as failures.

Knowledge lookups count each distinct entry one search returned. Both
`search_knowledge` and automatic knowledge recall record content-free
`knowledge.lookup` events with only the namespace and entry ids. Application tools
that search a store themselves can call
`await cayu.knowledge.usage.record_knowledge_lookup(namespace, entry_ids)` after a
search; outside a Runtime context it does nothing. Pass only returned ids, never
queries or text.

Each report names the same `store_id` as the registration and is counted against
that store's cursor, so every event is counted once across retries, restarts,
several processes, rollouts, and session stores that start over or run side by
side. When Cloud answers 409:

- `usage_cursor`: another process moved the cursor; the reporter counts again
  from `detail.cursor`.
- `store_not_registered`: Cloud keeps no cursor for this store (it drops one
  silent for 30 days); the reporter registers again, starting where its reports
  reached.
- `not_registered`: the reporter registers again.
- `release_not_current`: a newer Release counts the Agent's use; the reporter
  backs off and tries again later without registering again.

A database restored from a backup keeps its `store_id`, so Cloud's cursor can be
past the store's newest event. When the reporter finds nothing to report and
Cloud's cursor is beyond the newest event in its store (read after Cloud answered
it), it registers again with `replaces_cursor` set to that cursor and
`start_cursor` at its newest event, and Cloud moves the cursor back if it still
holds that value and hasn't replaced it before. The same covers a store without an identity that started over
below Cloud's cursor. A store without an identity that starts over above Cloud's
cursor can't be told apart from the old one; migrate it (`cayu storage migrate`)
to count it exactly.

Retained history bounds what can still be counted: deleting sessions or events
before they are reported removes their use.
