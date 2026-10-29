import assert from "node:assert/strict"
import test from "node:test"
import { validateUsageRollupSearch } from "../src/lib/usage-rollup-search.ts"
import { RANGE_CHOICES, rangeOf, resolveWindow } from "../src/next/routes/usage/usage-window.ts"

const now = Date.parse("2026-10-29T12:34:56.789Z")

test("Usage keeps the production default and explicit 30-day bookmarks", () => {
  for (const search of [{}, { range: "30d" }, { range: "constructor" }]) {
    assert.equal(rangeOf(validateUsageRollupSearch(search).range), "30d")
  }
  assert.deepEqual(
    RANGE_CHOICES.slice(0, 3).map((choice) => choice.label),
    ["Last 24 hours", "Last 7 days", "Last 30 days"],
  )
})

test("preset windows align the shared request span and choose a bounded series resolution", () => {
  for (const range of ["24h", "7d", "30d", "90d", "365d"]) {
    const window = resolveWindow(range, undefined, undefined, now)
    assert.ok(!("error" in window))
    assert.equal(window.startAt, new Date(window.start).toISOString())
    assert.equal(window.endAt, new Date(now).toISOString())
    assert.equal(new Date(window.start).getMinutes(), 0)
    if (range !== "24h") assert.equal(new Date(window.start).getHours(), 0)
    assert.equal(
      window.unit,
      range === "24h" ? "hour" : ["90d", "365d"].includes(range) ? "week" : "day",
    )
  }
})

test("custom windows preserve timezone and microsecond boundaries before display conversion", () => {
  const start = "2026-07-01T00:00:59.999123Z"
  const end = "2026-07-02T08:30:00.123456+06:00"
  const window = resolveWindow("custom", start, end, now)
  assert.ok(!("error" in window))
  assert.equal(window.startAt, start)
  assert.equal(window.endAt, end)
  const tiny = resolveWindow(
    "custom",
    "2026-07-01T00:00:00.000001Z",
    "2026-07-01T00:00:00.000002Z",
    now,
  )
  assert.ok(!("error" in tiny))
  assert.equal(tiny.startAt, "2026-07-01T00:00:00.000001Z")
  assert.equal(tiny.endAt, "2026-07-01T00:00:00.000002Z")
  assert.equal(tiny.unit, "hour")
})

test("custom windows reject calendar normalization, missing timezone and excess bounds", () => {
  for (const [start, end] of [
    ["2026-02-30T00:00:00Z", "2026-03-03T00:00:00Z"],
    ["2026-07-01T00:00:00", "2026-07-02T00:00:00Z"],
    ["2026-07-02T00:00:00Z", "2026-07-02T00:00:00Z"],
    ["2025-01-01T00:00:00Z", "2026-07-02T00:00:00Z"],
  ])
    assert.ok("error" in resolveWindow("custom", start, end, now), `${start} → ${end}`)
})

test("artifact URLs preserve exact identity and accept only valid paging and scope", async () => {
  const { validateArtifactSearch } = await import("../src/lib/artifact-search.ts")
  const exact = validateArtifactSearch({
    artifact: "file",
    artifact_store_id: "store-two",
    preview_store_id: "store-one",
    q: "report",
    offset: "25",
    scope: "environment",
  })
  assert.equal(exact.artifact, "file")
  assert.equal(exact.artifact_store_id, "store-two")
  assert.equal(exact.preview_store_id, "store-one")
  assert.equal(validateArtifactSearch({ preview_store_id: ["store"] }).preview_store_id, undefined)
  assert.equal(exact.q, "report")
  assert.equal(exact.offset, "25")
  assert.equal(exact.scope, "environment")
  for (const offset of [true, [], {}, "-1", "1.5", "Infinity", "9007199254740992"])
    assert.equal(validateArtifactSearch({ offset }).offset, undefined)
  assert.equal(
    validateArtifactSearch({ scope: "constructor", artifact: ["file"] }).scope,
    undefined,
  )
  assert.equal(validateArtifactSearch({ artifact: ["file"] }).artifact, undefined)
})

test("unknown environment component names stay readable instead of resolving prototype properties", async () => {
  const model = await import("../src/next/routes/environments/environment-model.ts")
  for (const name of ["constructor", "toString", "__proto__", "CustomRunner"]) {
    for (const [reader, field] of [
      [model.runsOn, "runner_type"],
      [model.knowledgeText, "knowledge_store_type"],
      [model.workspaceText, "binding_type"],
      [model.secretsText, "vault_type"],
      [model.networkText, "proxy_type"],
    ]) {
      assert.equal(
        typeof reader({ [field]: name, workspace_id: "workspace" }),
        "string",
        `${field}: ${name}`,
      )
    }
  }
})

test("observed branch handles retain terminal states and incomplete evidence", async () => {
  const { branchObservationText } = await import(
    "../src/next/routes/environments/environment-model.ts"
  )
  assert.match(
    branchObservationText({ attached_count: 1, statuses: ["committed"], truncated: false }),
    /1 committed/,
  )
  assert.match(
    branchObservationText({
      attached_count: 2,
      statuses: ["active", "rolled_back"],
      truncated: false,
    }),
    /1 active, 1 rolled back/,
  )
  assert.match(
    branchObservationText({ attached_count: 0, statuses: [], truncated: true }),
    /unavailable or incomplete/,
  )
  assert.match(
    branchObservationText({
      attached_count: 2,
      statuses: ["committed", "fenced"],
      truncated: true,
    }),
    /incomplete.*1 committed, 1 fenced/,
  )
  assert.match(
    branchObservationText({ attached_count: 0, statuses: [], truncated: false }),
    /No branch handles are attached here/,
  )
})

test("mutation-only System surfaces use mutation evidence while read-only states remain explicit", async () => {
  const { turnedOff } = await import("../src/next/routes/system/system-labels.ts")
  const enabled = { enabled: true }
  const unsupported = { enabled: false, unavailable_reason: "unsupported" }
  const missing = { enabled: false, unavailable_reason: "not_configured" }
  const diagnostics = (surfaces) => ({ capabilities: { surfaces, mutations: {} } })
  assert.deepEqual(
    turnedOff(diagnostics({ task_graph_creation: { read: unsupported, mutate: enabled } })),
    [],
  )
  assert.match(
    turnedOff(diagnostics({ task_graph_creation: { read: unsupported, mutate: missing } }))[0].why,
    /Not set up/,
  )
  assert.match(
    turnedOff(diagnostics({ tasks: { read: enabled, mutate: unsupported } }))[0].name,
    /read-only/,
  )
  assert.match(
    turnedOff(diagnostics({ artifacts: { read: missing, mutate: unsupported } }))[0].why,
    /No artifact store/,
  )
})
