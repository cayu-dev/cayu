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
