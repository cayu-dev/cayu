import assert from "node:assert/strict"
import test from "node:test"
import { fileURLToPath } from "node:url"
import { rolldown } from "rolldown"

const builder = await rolldown({
  cwd: fileURLToPath(new URL("..", import.meta.url)),
  input: "usage-cost-fixture",
  platform: "node",
  plugins: [
    {
      name: "focused-fixture",
      resolveId(id) {
        if (id === "@prototype-seeds") return "\0empty-seeds"
        if (id === "usage-cost-fixture") return "\0usage-cost-fixture"
      },
      load(id) {
        if (id === "\0empty-seeds") return "export default []"
        if (id === "\0usage-cost-fixture")
          return [
            'export { usageCostText, validateCostRanking, rankingExplanation } from "./src/next/routes/usage/usage-costs.ts"',
            'export { QueryClient } from "@tanstack/react-query"',
            'export { readUsage } from "./src/next/routes/usage/usage-query.ts"',
            'export { createMockApi } from "./src/next/mock/api.ts"',
            'export { coreRoutes } from "./src/next/mock/legacy/core.ts"',
          ].join("\n")
      },
    },
  ],
})
const bundled = await builder.generate({ format: "esm", codeSplitting: false })
await builder.close()
globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/product/api" } }
const {
  QueryClient,
  readUsage,
  createMockApi,
  coreRoutes,
  usageCostText,
  validateCostRanking,
  rankingExplanation,
} = await import(
  `data:text/javascript;base64,${Buffer.from(bundled.output[0].code).toString("base64")}`
)

test("prototype costs rank the entire scope before slicing and preserve grouped remainders", () => {
  const api = createMockApi()
  const seed = api.records()
  const source = structuredClone(seed.sessions[0])
  seed.sessions = Array.from({ length: 12 }, (_, i) => {
    const row = structuredClone(source)
    row.session.id = `cost-${i}`
    row.session.agent_name = "writer"
    row.session.provider_name = "fake"
    row.session.model = i === 11 ? "expensive" : "cheap"
    row.overview.title = `Report ${i}`
    row.overview.cost = { amount: i === 11 ? 0.01 : 0.001, currency: "USD" }
    row.usage.model_steps = 1
    row.usage.usage.input_tokens = i === 11 ? "10" : "1000"
    row.usage.usage.output_tokens = "0"
    row.usage.usage.total_tokens = row.usage.usage.input_tokens
    row.events = [{ type: "model.completed", timestamp: seed.now }]
    return row
  })
  const route = coreRoutes.find((route) => route.path === "/usage/rollup")
  const request = {
    start_at: new Date(api.now() - 86400000).toISOString(),
    end_at: new Date(api.now() + 1000).toISOString(),
    session_group_limit: 1,
    group_limit: 1,
    session_order: "cost",
  }
  const read = () => route.handle({ body: request }, { records: () => seed, now: api.now })
  let data = read()
  assert.equal(data.session_ranking.effective_order, "cost")
  assert.equal(data.session_breakdown.groups[0].session_id, "cost-11")
  assert.equal(data.session_breakdown.groups[0].title, "Report 11")
  assert.equal(data.session_breakdown.groups[0].agent_name, "writer")
  assert.equal(data.session_breakdown.remainder.group_count, "11")
  assert.equal(data.session_breakdown.accuracy.kind, "truncated")
  assert.equal(data.model_cost_breakdown.remainder.group_count, "1")
  assert.equal(data.model_cost_breakdown.remainder.cost.currencies[0].total_cost, "0.010000")
  assert.equal(data.provider_cost_breakdown.groups[0].cost.currencies[0].total_cost, "0.021000")
  seed.sessions[0].overview.cost.currency = "EUR"
  data = read()
  assert.equal(data.session_ranking.effective_order, "tokens")
  assert.equal(data.session_ranking.unavailable_reason, "mixed_currencies")
  seed.sessions[0].overview.cost = null
  data = read()
  assert.equal(data.session_ranking.unavailable_reason, "prices_incomplete")
  for (const row of seed.sessions) row.overview.cost = null
  data = read()
  assert.equal(data.session_ranking.unavailable_reason, "pricing_not_configured")
  assert.equal(data.provider_cost_breakdown, null)
})

const exactCost = (currencies = []) => ({
  accuracy: { kind: "exact" },
  currencies,
  unpriced_model_steps: "0",
  unevaluated_model_steps: "0",
  unpriced_reasons: [],
})
test("cost presentation preserves exact currencies, zero, unknown and incomplete coverage", () => {
  assert.equal(usageCostText(null), "Unavailable")
  assert.equal(usageCostText(exactCost()), "0 (no priced usage)")
  const cost = exactCost([
    { currency: "USD", total_cost: "12345678901234567890.12" },
    { currency: "EUR", total_cost: "0.000001" },
  ])
  const text = usageCostText(cost)
  assert.match(text, /12,345,678,901,234,567,890.12/)
  assert.match(text, /USD/)
  assert.match(text, /EUR/)
  assert.match(text, /<EUR\s0.0001/)
  assert.equal(text.includes("partial"), false)
  for (const uncertainty of [
    { accuracy: { kind: "sampled" } },
    { accuracy: { kind: "truncated" } },
    { unpriced_model_steps: "1" },
    { unevaluated_model_steps: "1" },
    { unpriced_auxiliary_attempts: "1" },
    { unpriced_reasons: [{ reason: "Hosted usage has no price", model_steps: "0" }] },
  ]) {
    assert.match(usageCostText({ ...cost, ...uncertainty }), /partial/)
    assert.equal(usageCostText({ ...exactCost(), ...uncertainty }), "Unavailable")
  }
})
test("cost requests require truthful order metadata and explain every fallback", () => {
  const response = (order, reason) => ({
    session_breakdown: { order },
    session_ranking: {
      requested_order: "cost",
      effective_order: order,
      unavailable_reason: reason,
    },
  })
  assert.equal(validateCostRanking(response("cost", null)).session_ranking.effective_order, "cost")
  for (const reason of [
    "pricing_not_configured",
    "prices_incomplete",
    "mixed_currencies",
    "candidate_limit",
    "candidate_byte_limit",
    "pricing_projection_incomplete",
    "future_reason",
  ]) {
    const data = response("tokens", reason)
    assert.equal(validateCostRanking(data), data)
    assert.match(rankingExplanation(data, true), /[Tt]okens/)
  }
  for (const data of [
    {},
    response("tokens", null),
    response("cost", "prices_incomplete"),
    { ...response("cost", null), session_breakdown: { order: "tokens" } },
    response("future", null),
  ])
    assert.throws(() => validateCostRanking(data), /valid session cost ranking/)
  assert.match(rankingExplanation({}, false), /not supported/)
})

test("prototype currency evidence excludes no-window and tool-only sessions but keeps real zero work", () => {
  const api = createMockApi()
  const seed = api.records()
  const source = structuredClone(seed.sessions[0])
  seed.sessions = ["historical", "tool-only", "zero-model"].map((name) => {
    const row = structuredClone(source)
    row.session.id = name
    row.overview.cost = {
      amount: name === "zero-model" ? 0 : 1,
      currency: name === "historical" ? "EUR" : "USD",
    }
    row.events = [
      {
        type: name === "tool-only" ? "tool.call.started" : "model.completed",
        timestamp:
          name === "historical" ? new Date(api.now() - 2 * 86400000).toISOString() : seed.now,
      },
    ]
    return row
  })
  const route = coreRoutes.find((route) => route.path === "/usage/rollup")
  const body = {
    start_at: new Date(api.now() - 86400000).toISOString(),
    end_at: new Date(api.now() + 1000).toISOString(),
    session_group_limit: 3,
    session_order: "cost",
  }
  const data = route.handle({ body }, { records: () => seed, now: api.now })
  assert.equal(data.session_breakdown.groups.length, 3)
  for (const group of data.session_cost_breakdown.groups) {
    if (group.session_id !== "zero-model") assert.deepEqual(group.cost.currencies, [])
    else assert.equal(group.cost.currencies[0].total_cost, "0.000000")
  }
  const limited = route.handle(
    { body: { ...body, session_group_limit: 1 } },
    { records: () => seed, now: api.now },
  )
  for (const summary of [
    ...limited.session_cost_breakdown.groups.map((g) => g.cost),
    limited.session_cost_breakdown.remainder.cost,
  ])
    assert.ok(summary.currencies.every((c) => BigInt(c.model_steps) > 0n))
})

test("Usage query owner discards every scope on denial and fences pre-denial late reads", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const first = ["usage", "rollup", "first"]
  const other = ["usage", "rollup", "other"]
  const counts = ["usage", "counts", "earlier-window"]
  const inventory = ["usage", "environment-options"]
  const load = (key, read) =>
    client.fetchQuery({
      queryKey: key,
      queryFn: ({ signal }) => readUsage(client, key, signal, read),
    })
  await load(first, async () => "protected report")
  await load(other, async () => "other report")
  await load(counts, async () => "earlier histogram")
  client.setQueryData(inventory, "unrelated inventory")
  const transient = Object.assign(new Error("Unavailable"), { status: 503 })
  await assert.rejects(
    load(first, async () => {
      throw transient
    }),
  )
  assert.equal(client.getQueryData(first).value, "protected report")
  let release
  const late = new Promise((resolve) => {
    release = resolve
  })
  const pending = load(other, () => late).catch(() => null)
  const denial = Object.assign(new Error("Denied"), { status: 403 })
  await load(first, async () => {
    throw denial
  })
  for (const key of [first, other, counts]) assert.equal(client.getQueryData(key).value, null)
  assert.equal(client.getQueryData(inventory), "unrelated inventory")
  release("pre-denial delayed report")
  await pending
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(client.getQueryData(other).value, null)
  await assert.rejects(
    load(other, async () => {
      throw transient
    }),
  )
  assert.equal(client.getQueryData(other).value, null)
  await load(other, async () => "fresh authorized report")
  assert.equal(client.getQueryData(other).value, "fresh authorized report")
  assert.equal(client.getQueryData(first).value, null)
  client.clear()
})
