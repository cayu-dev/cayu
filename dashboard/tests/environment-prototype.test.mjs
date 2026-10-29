import assert from "node:assert/strict"
import test from "node:test"
import { fileURLToPath } from "node:url"
import { rolldown } from "rolldown"

const builder = await rolldown({
  cwd: fileURLToPath(new URL("..", import.meta.url)),
  input: "environment-fixture",
  platform: "node",
  plugins: [
    {
      name: "focused-environment-fixture",
      resolveId(id) {
        if (id === "@prototype-seeds") return "\0empty-seeds"
        if (id === "environment-fixture") return "\0environment-fixture"
      },
      load(id) {
        if (id === "\0empty-seeds") return "export default []"
        if (id === "\0environment-fixture")
          return [
            'export { createMockApi } from "./src/next/mock/api.ts"',
            'export { coreRoutes } from "./src/next/mock/legacy/core.ts"',
          ].join("\n")
      },
    },
  ],
})
const bundled = await builder.generate({ format: "esm", codeSplitting: false })
await builder.close()
globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }
const { createMockApi, coreRoutes } = await import(
  `data:text/javascript;base64,${Buffer.from(bundled.output[0].code).toString("base64")}`
)

test("prototype inventory counts every retained session and reports durable activity and declarations", () => {
  const api = createMockApi()
  const seed = structuredClone(api.records())
  const initial = seed.sessions[0]
  seed.sessions = [0, 1, 2].map((index) => ({
    ...structuredClone(initial),
    session: {
      ...initial.session,
      id: `retained-${index}`,
      environment_name: "checkout",
      updated_at: "2030-01-01T00:00:00Z",
    },
    overview: {
      ...initial.overview,
      last_activity_at: `2026-10-${20 + index}T00:00:00Z`,
      debug: { ...initial.overview.debug, is_eval: index === 1, is_fork: index === 2 },
    },
  }))
  const now = Date.parse("2026-10-23T00:00:00Z")
  const route = coreRoutes.find((route) => route.path === "/environments")
  const page = route.handle({}, { records: () => seed, now: () => now })
  const checkout = page.environments.find((row) => row.name === "checkout")
  assert.equal(checkout.session_count, "3")
  assert.equal(checkout.last_activity_at, "2026-10-22T00:00:00Z")
  assert.equal(checkout.activity_as_of, new Date(now).toISOString())
  assert.equal(checkout.artifact_store_kind, "local_disk")
  assert.equal(checkout.mcp_server_names.length, checkout.mcp_server_count)
  const sandbox = page.environments.find((row) => row.name === "sandbox")
  assert.equal(sandbox.session_count, "0")
  assert.equal(sandbox.last_activity_at, null)
  assert.equal(sandbox.artifact_store_kind, "s3")
  assert.equal(sandbox.mcp_server_names.length, 3)
})
