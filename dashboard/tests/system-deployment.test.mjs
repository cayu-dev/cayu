import assert from "node:assert/strict"
import test from "node:test"
import { fileURLToPath } from "node:url"
import { rolldown } from "rolldown"

const builder = await rolldown({
  cwd: fileURLToPath(new URL("..", import.meta.url)),
  input: "system-fixture",
  platform: "node",
  plugins: [
    {
      name: "focused-system-fixture",
      resolveId(id) {
        if (id === "@prototype-seeds") return "\0empty-seeds"
        if (id === "system-fixture") return "\0system-fixture"
      },
      load(id) {
        if (id === "\0empty-seeds") return "export default []"
        if (id === "\0system-fixture")
          return [
            'export { createMockApi } from "./src/next/mock/api.ts"',
            'export { coreRoutes } from "./src/next/mock/legacy/core.ts"',
            'export { deploymentKindLabel } from "./src/next/routes/system/system-labels.ts"',
          ].join("\n")
      },
    },
  ],
})
const bundled = await builder.generate({ format: "esm", codeSplitting: false })
await builder.close()
globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }
const { createMockApi, coreRoutes, deploymentKindLabel } = await import(
  `data:text/javascript;base64,${Buffer.from(bundled.output[0].code).toString("base64")}`
)

test("prototype explicitly declares local deployment without naming inference", () => {
  const seed = createMockApi().records()
  seed.app.name = "Cloud-looking fake name"
  const route = coreRoutes.find((route) => route.path === "/system/diagnostics")
  const result = route.handle(
    {},
    { records: () => seed, now: () => Date.parse("2026-10-30T00:00:00Z") },
  )
  assert.equal(result.deployment.kind, "local")
  assert.equal(result.capabilities.surfaces.deployment_kind.read.enabled, true)
})

test("deployment labels require an explicit supported kind", () => {
  for (const kind of ["local", "cloud", "embedded"]) {
    assert.equal(deploymentKindLabel(kind), kind.charAt(0).toUpperCase() + kind.slice(1))
  }
  for (const kind of [undefined, null, true, 1, "Cloud", "unknown", "constructor"]) {
    assert.equal(deploymentKindLabel(kind), "Not reported")
  }
})
