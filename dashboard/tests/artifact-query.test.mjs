import assert from "node:assert/strict"
import test from "node:test"
import { fileURLToPath } from "node:url"
import { rolldown } from "rolldown"

const builder = await rolldown({
  cwd: fileURLToPath(new URL("..", import.meta.url)),
  input: "artifact-fixture",
  platform: "node",
  plugins: [
    {
      name: "focused-artifact-fixture",
      resolveId(id) {
        if (id === "@prototype-seeds") return "\0empty-seeds"
        if (id === "artifact-fixture") return "\0artifact-fixture"
      },
      load(id) {
        if (id === "\0empty-seeds") return "export default []"
        if (id === "\0artifact-fixture")
          return [
            'export { createMockApi } from "./src/next/mock/api.ts"',
            'export { coreRoutes } from "./src/next/mock/legacy/core.ts"',
            'export { readArtifactList } from "./src/lib/artifact-query.ts"',
            'export { QueryClient } from "@tanstack/react-query"',
            'export { validateArtifactSearch } from "./src/lib/artifact-search.ts"',
          ].join("\n")
      },
    },
  ],
})
const bundled = await builder.generate({ format: "esm", codeSplitting: false })
await builder.close()
globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }
const { createMockApi, coreRoutes, readArtifactList, QueryClient, validateArtifactSearch } =
  await import(
    `data:text/javascript;base64,${Buffer.from(bundled.output[0].code).toString("base64")}`
  )

test("prototype searches producing titles and complete file inventory before sorting/paging", () => {
  const api = createMockApi()
  const seed = structuredClone(api.records())
  const original = seed.sessions[0]
  seed.sessions = Array.from({ length: 30 }, (_, index) => ({
    ...structuredClone(original),
    session: { ...original.session, id: `finished-${index}` },
    overview: {
      ...original.overview,
      title: `Synthetic report ${index}`,
      state: "completed",
      ended_at: "2026-10-23T00:00:00Z",
    },
  }))
  const context = { records: () => seed, now: () => Date.parse("2026-10-23T00:00:00Z") }
  const route = coreRoutes.find((route) => route.path === "/artifacts")
  const read = (values) => route.handle({ query: new URLSearchParams(values) }, context)
  const all = read({ limit: "500", order_by: "filename_asc" })
  assert.ok(all.artifacts.length > 25)
  const last = all.artifacts.at(-1)
  const search = read({ q: last.filename, limit: "1" })
  assert.ok(search.artifacts.some((row) => row.id === last.id))
  const producer = all.artifacts.find((row) => row.session_id)
  assert.ok(producer.session_title)
  assert.ok(read({ q: producer.session_title }).artifacts.some((row) => row.id === producer.id))
  assert.deepEqual(
    read({ limit: "5", offset: "25", order_by: "filename_asc" }).artifacts,
    all.artifacts.slice(25, 30),
  )
  assert.deepEqual(
    read({ limit: "500", order_by: "filename_desc" }).artifacts,
    [...all.artifacts].reverse(),
  )
})

test("typed search keeps independent preview/store scope and accepts only documented sort names", () => {
  const query = validateArtifactSearch({
    q: " report ",
    order_by: "size_bytes_desc",
    artifact_store_id: "list",
    preview_store_id: "preview",
    artifact: "same",
    offset: "25",
  })
  assert.equal(query.order_by, "size_bytes_desc")
  assert.equal(query.artifact_store_id, "list")
  assert.equal(query.preview_store_id, "preview")
  assert.equal(query.offset, "25")
  for (const value of [true, null, {}, "newest"])
    assert.equal(validateArtifactSearch({ order_by: value }).order_by, undefined)
})

test("artifact denial fences all scopes, ignored-abort late reads, and later transient errors", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const first = ["artifact-list", { scope: "session" }]
  const second = ["artifact-list", { scope: "environment" }]
  const preview = ["artifact-preview", "same", { artifact_store_id: "one" }]
  const other = ["not-artifacts"]
  const page = { artifacts: [{ filename: "protected" }], total_count: 1 }
  const fetch = (key, read) =>
    client.fetchQuery({
      queryKey: key,
      queryFn: ({ signal }) => readArtifactList(client, key, signal, read),
    })
  await fetch(first, async () => page)
  await fetch(second, async () => page)
  await fetch(preview, async () => ({
    artifact: page.artifacts[0],
    text_preview: "protected contents",
  }))
  client.setQueryData(other, "unrelated")
  let release
  const blocked = new Promise((resolve) => {
    release = resolve
  })
  const pending = fetch(second, () => blocked).catch((error) => error)
  const denied = Object.assign(new Error("Forbidden"), { status: 403 })
  await fetch(first, async () => {
    throw denied
  })
  assert.equal(client.getQueryData(first).value, null)
  assert.equal(client.getQueryData(second).value, null)
  release(page)
  await pending
  await assert.rejects(
    fetch(first, async () => {
      throw new Error("Unavailable")
    }),
    /Unavailable/,
  )
  assert.equal(client.getQueryData(first).value, null)
  assert.equal(client.getQueryData(second).value, null)
  assert.equal(client.getQueryData(preview).value, null)
  assert.equal(client.getQueryData(other), "unrelated")
  await fetch(first, async () => page)
  assert.deepEqual(client.getQueryData(first).value, page)
  assert.equal(client.getQueryData(second).value, null)
  client.clear()
})
