import assert from "node:assert/strict"
import test from "node:test"
import { fileURLToPath } from "node:url"
import { rolldown } from "rolldown"

const builder = await rolldown({
  cwd: fileURLToPath(new URL("..", import.meta.url)),
  input: "usage-series-fixture",
  platform: "node",
  plugins: [
    {
      name: "focused-fixture",
      resolveId(id) {
        if (id === "@prototype-seeds") return "\0empty-seeds"
        if (id === "usage-series-fixture") return "\0usage-series-fixture"
      },
      load(id) {
        if (id === "\0empty-seeds") return "export default []"
        if (id === "\0usage-series-fixture")
          return [
            'export { realApi } from "./src/next/api/real.ts"',
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
const { realApi, createMockApi, coreRoutes } = await import(
  `data:text/javascript;base64,${Buffer.from(bundled.output[0].code).toString("base64")}`
)

test("batched histogram adapter carries exact filters and real AbortSignal", async () => {
  const original = globalThis.fetch
  const controller = new AbortController()
  let request
  globalThis.fetch = (url, init) => {
    request = { url, init }
    return new Promise((_resolve, reject) =>
      init.signal?.addEventListener("abort", () => reject(init.signal.reason), { once: true }),
    )
  }
  try {
    const pending = realApi.overviewCounts(
      {
        since: "2026-01-01T00:00:00.000001+06:00",
        until: "2026-01-01T00:00:00.000002+06:00",
        bucket_unit: "hour",
        timezone: "Asia/Bishkek",
        environment_name: "Review / λ",
        agent_name: "writer",
        provider_name: "p",
        model: "m",
      },
      controller.signal,
    )
    assert.equal(request.init.signal, controller.signal)
    const url = new URL(request.url, "https://example.test")
    assert.equal(url.pathname, "/product/api/sessions/overview/counts")
    assert.equal(url.searchParams.get("since"), "2026-01-01T00:00:00.000001+06:00")
    assert.equal(url.searchParams.get("environment_name"), "Review / λ")
    assert.equal(url.searchParams.get("timezone"), "Asia/Bishkek")
    controller.abort()
    await assert.rejects(pending, { name: "AbortError" })
  } finally {
    globalThis.fetch = original
  }
})

test("prototype rollup and filtered start histogram have matching buckets and additive activity", async () => {
  const api = createMockApi()
  const end = api.now() + 1000
  const start = end - 30 * 86400000
  const query = {
    since: new Date(start).toISOString(),
    until: new Date(end).toISOString(),
    bucket_unit: "day",
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  }
  const rollup = coreRoutes.find((route) => route.path === "/usage/rollup")
  for (const filter of [
    {},
    { agent_name: api.records().sessions[0].session.agent_name },
    { environment_name: "no-such-environment" },
    { provider_name: "no-such-provider" },
    { model: "no-such-model" },
  ]) {
    const counts = await api.overviewCounts({ ...query, ...filter })
    const data = rollup.handle(
      {
        body: {
          start_at: query.since,
          end_at: query.until,
          series: { unit: "day", timezone: query.timezone },
          session_filter: filter,
        },
      },
      { records: api.records, now: api.now },
    )
    assert.ok(counts.series && data.series)
    assert.deepEqual(
      counts.series.buckets.map((b) => [b.start_at, b.end_at]),
      data.series.buckets.map((b) => [b.start_at, b.end_at]),
    )
    assert.equal(
      counts.series.buckets.reduce((total, b) => total + BigInt(b.total_count), 0n),
      BigInt(counts.total_count),
    )
    for (const field of ["model_steps", "tool_calls"])
      assert.equal(
        data.series.buckets.reduce((total, b) => total + BigInt(b.totals[field]), 0n),
        BigInt(data.totals[field]),
      )
    assert.equal(
      data.series.buckets.reduce((total, b) => total + BigInt(b.totals.usage.total_tokens), 0n),
      BigInt(data.totals.usage.total_tokens),
    )
  }
})

test("synthetic activity counters conserve token remainders across periods", () => {
  const api = createMockApi()
  const seed = structuredClone(api.records())
  const row = seed.sessions[0]
  seed.sessions = [row]
  row.usage.usage.input_tokens = "5"
  row.usage.usage.output_tokens = "2"
  row.usage.usage.total_tokens = "7"
  row.events = [0, 1, 2].map((i) => ({
    type: "model.completed",
    timestamp: new Date(Date.UTC(2026, 0, 1, i)).toISOString(),
  }))
  const data = coreRoutes
    .find((route) => route.path === "/usage/rollup")
    .handle(
      {
        body: {
          start_at: "2026-01-01T00:00:00Z",
          end_at: "2026-01-01T03:00:00Z",
          series: { unit: "hour", timezone: Intl.DateTimeFormat().resolvedOptions().timeZone },
        },
      },
      { records: () => seed, now: api.now },
    )
  assert.equal(
    data.series.buckets.reduce((n, b) => n + BigInt(b.totals.usage.total_tokens), 0n),
    BigInt(data.totals.usage.total_tokens),
  )
})

test("prototype series preserve valid sub-millisecond request windows", async () => {
  const api = createMockApi()
  const start_at = "2026-01-01T00:00:00.000001Z"
  const end_at = "2026-01-01T00:00:00.000002Z"
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone
  const data = coreRoutes
    .find((route) => route.path === "/usage/rollup")
    .handle(
      { body: { start_at, end_at, series: { unit: "hour", timezone } } },
      { records: api.records, now: api.now },
    )
  const counts = await api.overviewCounts({
    since: start_at,
    until: end_at,
    bucket_unit: "hour",
    timezone,
  })
  assert.equal(data.start_at, start_at)
  assert.equal(data.end_at, end_at)
  assert.deepEqual(
    data.series.buckets.map((b) => [b.start_at, b.end_at]),
    [[start_at, end_at]],
  )
  assert.deepEqual(
    counts.series.buckets.map((b) => [b.start_at, b.end_at]),
    [[start_at, end_at]],
  )
})

test("prototype start histogram uses session creation, not a later run start", async () => {
  const api = createMockApi()
  const rows = api.records().sessions
  rows.splice(1)
  const now = api.now()
  rows[0].session.created_at = new Date(now - 10 * 86400000).toISOString()
  rows[0].overview.started_at = new Date(now).toISOString()
  const result = await api.overviewCounts({
    since: new Date(now - 3600000).toISOString(),
    until: new Date(now + 1000).toISOString(),
    bucket_unit: "hour",
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  })
  assert.equal(result.total_count, "0")
  assert.ok(result.series.buckets.every((bucket) => bucket.total_count === "0"))
})

test("prototype half-open membership compares exact instants for starts and events", async () => {
  for (const [start_at, end_at, instants] of [
    [
      "2026-01-01T00:00:00.000001Z",
      "2026-01-01T00:00:00.001000Z",
      [
        "2026-01-01T00:00:00.000000Z",
        "2026-01-01T06:00:00.000001+06:00",
        "2026-01-01T00:00:00.000999Z",
        "2026-01-01T00:00:00.001000Z",
      ],
    ],
    [
      "2026-01-01T00:00:00.000001Z",
      "2026-01-01T00:00:00.000003Z",
      [
        "2026-01-01T00:00:00.000000Z",
        "2026-01-01T06:00:00.000001+06:00",
        "2026-01-01T00:00:00.000002Z",
        "2026-01-01T00:00:00.000003Z",
      ],
    ],
    [
      "2026-01-01T00:00:00.001Z",
      "2026-01-01T00:00:00.003Z",
      [
        "2026-01-01T00:00:00.000Z",
        "2026-01-01T06:00:00.001+06:00",
        "2026-01-01T00:00:00.002Z",
        "2026-01-01T00:00:00.003Z",
      ],
    ],
  ]) {
    const api = createMockApi()
    const seed = api.records()
    const source = seed.sessions[0]
    seed.sessions = instants.map((instant, index) => {
      const row = structuredClone(source)
      row.session.agent_name = `boundary-${index}`
      row.session.created_at = instant
      row.overview.forked = false
      row.metadata = {}
      row.events = ["model.completed", "tool.call.started"].map((type) => ({
        type,
        timestamp: instant,
      }))
      row.usage.usage.input_tokens = "5"
      row.usage.usage.output_tokens = "2"
      row.usage.usage.total_tokens = "7"
      return row
    })
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone
    for (let index = 0; index < 4; index++) {
      const expected = index === 1 || index === 2 ? "1" : "0"
      const agent_name = `boundary-${index}`
      const counts = await api.overviewCounts({
        since: start_at,
        until: end_at,
        agent_name,
        bucket_unit: "hour",
        timezone,
      })
      const data = coreRoutes
        .find((route) => route.path === "/usage/rollup")
        .handle(
          {
            body: {
              start_at,
              end_at,
              session_filter: { agent_name },
              series: { unit: "hour", timezone },
            },
          },
          { records: api.records, now: api.now },
        )
      assert.equal(counts.total_count, expected, `start membership ${instants[index]}`)
      assert.equal(counts.series.buckets[0].total_count, expected)
      for (const totals of [data.totals, data.series.buckets[0].totals]) {
        assert.equal(totals.model_steps, expected, `model membership ${instants[index]}`)
        assert.equal(totals.tool_calls, expected, `tool membership ${instants[index]}`)
        assert.equal(totals.usage.total_tokens, expected === "1" ? "7" : "0")
      }
    }
  }
})
