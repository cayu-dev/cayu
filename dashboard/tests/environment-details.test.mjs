import assert from "node:assert/strict"
import test from "node:test"
import { applyViewerAccess } from "../src/lib/dashboard-access.ts"
import { validateEnvironmentSearch } from "../src/lib/environment-search.ts"
import * as model from "../src/next/routes/environments/environment-model.ts"

const observation = {
  session_count: "9007199254740993",
  last_activity_at: "2026-10-30T01:02:03Z",
  activity_as_of: "2026-10-30T02:00:00Z",
}

test("environment activity needs capability and a complete native observation", () => {
  assert.deepEqual(model.environmentActivity(observation, true), {
    count: observation.session_count,
    last: observation.last_activity_at,
    asOf: observation.activity_as_of,
  })
  assert.equal(model.environmentActivity(observation, false), null)
  for (const invalid of [
    {},
    { ...observation, session_count: null },
    { ...observation, session_count: true },
    { ...observation, session_count: "-1" },
    { ...observation, session_count: "1.5" },
    { ...observation, activity_as_of: null },
    { ...observation, activity_as_of: "bad" },
    { ...observation, last_activity_at: null },
    { ...observation, session_count: "0" },
  ])
    assert.equal(model.environmentActivity(invalid, true), null)
  assert.deepEqual(
    model.environmentActivity({ ...observation, session_count: "0", last_activity_at: null }, true),
    {
      count: "0",
      last: null,
      asOf: observation.activity_as_of,
    },
  )
})

test("storage labels use declared kinds and distinguish no store from unknown metadata", () => {
  assert.equal(model.artifactStorageText({ artifact_store_id: null }), "No file storage")
  for (const [kind, expected] of [
    ["local_disk", "Local disk"],
    ["s3", "S3"],
    ["unknown", "Storage kind unavailable"],
    [null, "Storage kind unavailable"],
    ["future", "Storage kind unavailable"],
  ]) {
    assert.equal(
      model.artifactStorageText({
        artifact_store_id: "s3-not-authority",
        artifact_store_kind: kind,
      }),
      expected,
    )
  }
})

test("MCP names distinguish empty declarations from unavailable names", () => {
  assert.equal(model.mcpNamesText({ mcp_server_count: 0, mcp_server_names: [] }), "None")
  assert.equal(
    model.mcpNamesText({ mcp_server_count: 2, mcp_server_names: ["Lookup", "[REDACTED]"] }),
    "Lookup, [REDACTED]",
  )
  assert.equal(model.mcpNamesText({ mcp_server_count: 2 }), "2 MCP servers (names unavailable)")
})

test("members can read environment activity but gain no mutation authority", () => {
  const identity = { name: "Demo", runtime_version: "0.16.1", role: "member", full_access: false }
  const operation = { enabled: true, unavailable_reason: null }
  assert.equal(
    applyViewerAccess(operation, { kind: "surface", surface: "environment_details" }, identity)
      .enabled,
    true,
  )
  assert.equal(
    applyViewerAccess(
      operation,
      { kind: "surface", surface: "environment_details", operation: "mutate" },
      identity,
    ).enabled,
    false,
  )
})

test("dot environment names have distinct browser-safe selectors and ordinary URLs stay unchanged", () => {
  for (const name of [".", ".."]) {
    assert.deepEqual(model.environmentRoute(name), { to: "/environments", search: { name } })
    assert.deepEqual(validateEnvironmentSearch({ name, q: "existing" }), { name, q: "existing" })
  }
  for (const name of ["local", "%2E", "%2E%2E", "Review / λ space"])
    assert.deepEqual(model.environmentRoute(name), { to: "/environments/$name", params: { name } })
  for (const name of ["other", true, ["."], null])
    assert.equal(validateEnvironmentSearch({ name }).name, undefined)
})

const caps = {
  isolation: false,
  net_changes: false,
  publication: "unsupported",
  recovery: "unsupported",
  retention: "unsupported",
  lifecycle_inspection: "unsupported",
  detail_code: "workspace_branching_unsupported",
}
const declaredNothing = {
  name: "checkout",
  is_factory: false,
  factory_type: null,
  workspace_id: null,
  workspace_branch_capabilities: caps,
  artifact_store_id: null,
  artifact_store_kind: null,
  runner_type: null,
  binding_type: null,
  vault_type: null,
  proxy_type: null,
  knowledge_store_type: null,
  mcp_server_count: 0,
  mcp_server_names: [],
}
const factory = { ...declaredNothing, is_factory: true, factory_type: "CheckoutEnvironmentFactory" }

test("a factory environment's parts are decided per session, not missing", () => {
  const decided = model.DECIDED_PER_SESSION
  assert.equal(model.setupText(factory), "Set up for each session by CheckoutEnvironmentFactory")
  assert.equal(model.setupText({ ...factory, factory_type: null }), "Set up for each session")
  assert.equal(model.runsOn(factory), "Set up for each session")
  assert.equal(model.runsOnShort(factory), "By CheckoutEnvironmentFactory")
  assert.equal(model.runsOnShort({ ...factory, factory_type: null }), "By an environment factory")
  assert.match(model.runsOnDetail(factory), /^CheckoutEnvironmentFactory creates this environment/)
  assert.doesNotMatch(model.runsOnDetail(factory), /aren't available/)
  assert.equal(model.workspaceLabel(factory), decided)
  assert.equal(model.workspaceLabel(factory, "No workspace"), decided)
  assert.equal(model.workspaceText(factory), null)
  assert.match(model.branchDetails(factory)[0], /^Decided when a session starts/)
  assert.equal(model.artifactStorageText(factory), decided)
  assert.match(model.noFileStoreText(factory), /^Decided when a session starts\./)
  assert.equal(model.knowledgeText(factory), decided)
  assert.match(model.knowledgeDetail(factory), /^Decided when a session starts/)
  assert.equal(model.secretsText(factory), decided)
  assert.equal(model.networkText(factory), decided)
  assert.equal(model.mcpNamesText(factory), decided)
  assert.equal(model.mcpShort(factory), "MCP servers decided per session")
  assert.equal(model.brief(model.workspaceLabel(factory)), "Per session")
  assert.equal(model.brief(model.knowledgeText(declaredNothing)), "None")
})

test("a factory's declared parts still show, and registered environments keep their wording", () => {
  const declared = {
    ...factory,
    runner_type: "DockerRunner",
    workspace_id: "w",
    binding_type: "GitRepositoryBinding",
    artifact_store_id: "files",
    artifact_store_kind: "s3",
    knowledge_store_type: "SQLiteKnowledgeStore",
  }
  assert.equal(model.runsOn(declared), "Docker container")
  assert.equal(model.runsOnShort(declared), "Fresh for each session")
  assert.equal(model.workspaceLabel(declared), "Git repository")
  assert.equal(model.artifactStorageText(declared), "S3")
  assert.equal(model.knowledgeDetail(declared), "SQLite")

  assert.equal(model.setupText(declaredNothing), null)
  assert.equal(model.runsOn(declaredNothing), "No runner")
  assert.equal(model.runsOnShort(declaredNothing), "No shell, Git or search tools")
  assert.equal(
    model.runsOnDetail(declaredNothing),
    "Shell, Git and text-search tools aren't available. The agent's own tools still work.",
  )
  assert.equal(model.workspaceLabel(declaredNothing), "None")
  assert.equal(model.workspaceLabel(declaredNothing, "No workspace"), "No workspace")
  assert.deepEqual(model.branchDetails(declaredNothing), ["This environment has no workspace."])
  assert.equal(model.artifactStorageText(declaredNothing), "No file storage")
  assert.match(model.noFileStoreText(declaredNothing), /^Nowhere\./)
  assert.equal(model.knowledgeText(declaredNothing), "None")
  assert.match(model.knowledgeDetail(declaredNothing), /^No knowledge store\./)
  assert.equal(model.secretsText(declaredNothing), "None")
  assert.equal(model.networkText(declaredNothing), "Not managed by Cayu")
  assert.equal(model.mcpNamesText(declaredNothing), "None")
  assert.equal(model.mcpShort(declaredNothing), "No MCP servers")
  assert.equal(
    model.runsOnShort({ ...declaredNothing, runner_type: "LocalRunner" }),
    "Shared by all sessions",
  )
})

test("a session's binding event says what its environment had", () => {
  assert.equal(model.SESSION_SETUP_EVENT, "environment.binding.completed")
  assert.deepEqual(
    model.sessionSetupFacts({
      binding_type: "NativeBinding",
      has_configured_runner: true,
      has_bound_runner: true,
      bound_workspace_id: "session-1",
    }),
    {
      commands: "Had a runner, so shell, Git and text-search tools could run",
      workspace: "Folder",
    },
  )
  assert.deepEqual(
    model.sessionSetupFacts({ has_configured_runner: false, has_bound_runner: false }),
    {
      commands: "No runner, so shell, Git and text-search tools weren't available",
      workspace: "None",
    },
  )
  // A redacted binding is an opaque alias; only a plain class name is put into words.
  assert.equal(
    model.sessionSetupFacts({ binding_type: "alias:abc", source_workspace_id: "w" }).workspace,
    "Workspace",
  )
  assert.equal(
    model.sessionSetupFacts({ binding_type: "ScratchCopyBinding", configured_workspace_id: "w" })
      .workspace,
    "Scratch Copy",
  )
  // The binding can drop the configured runner; the bound result wins.
  assert.match(
    model.sessionSetupFacts({ has_configured_runner: true, has_bound_runner: false }).commands,
    /^No runner/,
  )
  assert.match(model.sessionSetupFacts({ has_configured_runner: true }).commands, /^Had a runner/)
  assert.equal(model.sessionSetupFacts({}).commands, "Not recorded")
  assert.equal(model.sessionSetupFacts(undefined), null)
  assert.equal(model.sessionSetupFacts(null), null)
})
