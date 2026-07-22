import assert from "node:assert/strict"
import test from "node:test"

import { parseDashboardSearch, stringifyDashboardSearch } from "../src/lib/search-params.ts"
import {
  buildUsageRollupRequest,
  multilineUsageFilters,
  resolveUsageWindow,
  usageRollupFilterCount,
  usageRollupRequestState,
  usageRollupSearchWithoutFilters,
  usageRollupSessionSearch,
  usageTimestampFromUtcInput,
  usageTimestampInputValue,
} from "../src/lib/usage-rollup.ts"
import { usageRollupRange, validateUsageRollupSearch } from "../src/lib/usage-rollup-search.ts"

const NOW = new Date("2026-07-21T12:00:00.000Z")

test("usage URL state preserves JSON-like identifiers and repeated filters", () => {
  const parsed = parseDashboardSearch(
    "?range=7d&agent_name=true&provider_name=123&model=1e3&environment_name=null&label=tenant%3Dacme&label=tier%3Dcritical&label_selector=region%20in%20%28us%2Ceu%29&label_selector=%21archived",
  )
  const search = validateUsageRollupSearch(parsed)

  assert.deepEqual(search, {
    range: "7d",
    start_at: undefined,
    end_at: undefined,
    agent_name: "true",
    provider_name: "123",
    model: "1e3",
    environment_name: "null",
    parent_session_id: undefined,
    causal_budget_id: undefined,
    label: ["tenant=acme", "tier=critical"],
    label_selector: ["region in (us,eu)", "!archived"],
  })

  const serialized = stringifyDashboardSearch(search)
  assert.deepEqual(new URLSearchParams(serialized).getAll("label"), [
    "tenant=acme",
    "tier=critical",
  ])
  assert.deepEqual(validateUsageRollupSearch(parseDashboardSearch(serialized)), search)
})

test("relative ranges resolve once against an explicit UTC anchor", () => {
  assert.equal(usageRollupRange(validateUsageRollupSearch({})), "30d")
  assert.deepEqual(resolveUsageWindow(validateUsageRollupSearch({ range: "24h" }), NOW), {
    startAt: "2026-07-20T12:00:00.000Z",
    endAt: "2026-07-21T12:00:00.000Z",
  })
  assert.deepEqual(resolveUsageWindow(validateUsageRollupSearch({ range: "365d" }), NOW), {
    startAt: "2025-07-21T12:00:00.000Z",
    endAt: "2026-07-21T12:00:00.000Z",
  })
})

test("custom windows preserve validated boundaries and reject unsafe widening", () => {
  const search = validateUsageRollupSearch({
    range: "custom",
    start_at: "2026-07-01T06:00:00+06:00",
    end_at: "2026-07-02T06:00:00+06:00",
  })
  assert.deepEqual(resolveUsageWindow(search, NOW), {
    startAt: "2026-07-01T06:00:00+06:00",
    endAt: "2026-07-02T06:00:00+06:00",
  })

  assert.match(
    usageRollupRequestState(validateUsageRollupSearch({ range: "custom", start_at: "2026-07-01" }))
      .error,
    /RFC 3339|required/,
  )
  assert.match(
    usageRollupRequestState(
      validateUsageRollupSearch({
        range: "custom",
        start_at: "2026-02-30T00:00:00Z",
        end_at: "2026-03-03T00:00:00Z",
      }),
    ).error,
    /not a valid timestamp/,
  )
  assert.throws(
    () =>
      resolveUsageWindow(
        validateUsageRollupSearch({
          range: "custom",
          start_at: "2025-01-01T00:00:00Z",
          end_at: "2026-07-21T00:00:00Z",
        }),
      ),
    /366 days/,
  )
})

test("custom window fields preserve untouched RFC 3339 boundaries exactly", () => {
  const preciseStart = "2026-07-01T00:00:59.999123Z"
  const offsetEnd = "2026-07-02T08:30:00.123456+06:00"
  const startInput = usageTimestampInputValue(preciseStart)
  const endInput = usageTimestampInputValue(offsetEnd)

  assert.equal(startInput, "2026-07-01T00:00:59.999")
  assert.equal(endInput, "2026-07-02T02:30:00.123")
  assert.equal(
    usageTimestampFromUtcInput(startInput, "Custom start", {
      originalValue: preciseStart,
      originalInputValue: startInput,
    }),
    preciseStart,
  )
  assert.equal(
    usageTimestampFromUtcInput(endInput, "Custom end", {
      originalValue: offsetEnd,
      originalInputValue: endInput,
    }),
    offsetEnd,
  )
})

test("custom usage requests retain sub-millisecond boundaries in the POST contract", () => {
  const startAt = "2026-07-01T00:00:59.999123Z"
  const endAt = "2026-07-02T08:30:00.123456+06:00"
  const request = buildUsageRollupRequest(
    validateUsageRollupSearch({
      range: "custom",
      start_at: startAt,
      end_at: endAt,
    }),
  )

  assert.equal(request.start_at, startAt)
  assert.equal(request.end_at, endAt)
})

test("custom usage requests reject precision finer than the server can represent", () => {
  const search = validateUsageRollupSearch({
    range: "custom",
    start_at: "2026-07-01T00:00:00.0000001Z",
    end_at: "2026-07-01T00:00:00.0000002Z",
  })

  assert.match(usageRollupRequestState(search).error, /at most 6 fractional digits/)
  assert.throws(() => buildUsageRollupRequest(search), /at most 6 fractional digits/)
  assert.equal(usageTimestampInputValue(search.start_at), "")
  assert.throws(
    () => usageTimestampFromUtcInput("2026-07-01T00:00:00.1234567", "Custom start"),
    /at most 6 fractional digits/,
  )
})

test("custom window ordering compares fractional instants beyond milliseconds", () => {
  const startAt = "2026-07-01T00:00:00.000001Z"
  const endAt = "2026-07-01T00:00:00.000002Z"
  assert.deepEqual(
    resolveUsageWindow(
      validateUsageRollupSearch({ range: "custom", start_at: startAt, end_at: endAt }),
    ),
    { startAt, endAt },
  )
  assert.throws(
    () =>
      resolveUsageWindow(
        validateUsageRollupSearch({ range: "custom", start_at: endAt, end_at: startAt }),
      ),
    /start must be before/,
  )
})

test("custom window limits compare exact fractional boundaries", () => {
  const startAt = "2025-01-01T00:00:00.000001Z"
  const exactEndAt = "2026-01-02T00:00:00.000001Z"
  assert.deepEqual(
    resolveUsageWindow(
      validateUsageRollupSearch({ range: "custom", start_at: startAt, end_at: exactEndAt }),
    ),
    { startAt, endAt: exactEndAt },
  )
  assert.throws(
    () =>
      resolveUsageWindow(
        validateUsageRollupSearch({
          range: "custom",
          start_at: startAt,
          end_at: "2026-01-02T00:00:00.000002Z",
        }),
      ),
    /366 days/,
  )
})

test("custom window fields use an unambiguous UTC editing contract across DST folds", () => {
  const daylightFold = "2026-11-01T01:30:00-04:00"
  const standardFold = "2026-11-01T01:30:00-05:00"

  assert.equal(usageTimestampInputValue(daylightFold), "2026-11-01T05:30:00.000")
  assert.equal(usageTimestampInputValue(standardFold), "2026-11-01T06:30:00.000")
  assert.equal(
    usageTimestampFromUtcInput("2026-11-01T06:31:02.345", "Custom start"),
    "2026-11-01T06:31:02.345Z",
  )
  assert.equal(
    usageTimestampFromUtcInput("2026-11-01T06:31", "Custom start"),
    "2026-11-01T06:31:00.000Z",
  )
  assert.throws(
    () => usageTimestampFromUtcInput("2026-02-30T00:00:00.000", "Custom start"),
    /not a valid timestamp/,
  )
})

test("the request maps every supported aggregate filter and selector", () => {
  const search = validateUsageRollupSearch({
    range: "7d",
    agent_name: "operator",
    provider_name: "anthropic",
    model: "claude-example",
    environment_name: "production",
    parent_session_id: "parent-session",
    causal_budget_id: "budget-1",
    label: ["tenant=acme", "tier=critical", "__proto__=preserved"],
    label_selector: [" region in (us, eu) ", "! archived", "stage != draft"],
  })

  const request = buildUsageRollupRequest(search, { now: NOW })
  assert.deepEqual(request, {
    start_at: "2026-07-14T12:00:00.000Z",
    end_at: "2026-07-21T12:00:00.000Z",
    session_filter: {
      agent_name: "operator",
      provider_name: "anthropic",
      model: "claude-example",
      environment_name: "production",
      parent_session_id: "parent-session",
      causal_budget_id: "budget-1",
      labels: { tenant: "acme", tier: "critical", ["__proto__"]: "preserved" },
      label_selectors: [
        { key: "region", operator: "in", values: ["us", "eu"] },
        { key: "archived", operator: "not_exists", values: [] },
        { key: "stage", operator: "not_in", values: ["draft"] },
      ],
    },
    group_limit: 20,
    pricing_input_limit: 1000,
    pricing: null,
  })
})

test("invalid filters fail locally instead of silently broadening the aggregate", () => {
  assert.match(
    usageRollupRequestState(validateUsageRollupSearch({ label: ["missing-separator"] })).error,
    /key=value/,
  )
  assert.match(
    usageRollupRequestState(validateUsageRollupSearch({ label: ["tenant=one", "tenant=two"] }))
      .error,
    /Duplicate label/,
  )
  assert.match(
    usageRollupRequestState(validateUsageRollupSearch({ label_selector: ["region in us,eu"] }))
      .error,
    /must use \(a,b\)/,
  )
})

test("range and filter updates preserve only their independent URL state", () => {
  const original = validateUsageRollupSearch({
    range: "7d",
    agent_name: "operator",
    label: ["tenant=acme"],
  })
  const custom = validateUsageRollupSearch({
    ...original,
    range: "custom",
    start_at: "2026-07-01T00:00:00Z",
    end_at: "2026-07-02T00:00:00Z",
  })
  assert.equal(custom.agent_name, "operator")
  assert.equal(custom.start_at, "2026-07-01T00:00:00Z")

  const filtered = validateUsageRollupSearch({ ...custom, provider_name: "openai" })
  assert.equal(filtered.start_at, custom.start_at)
  assert.equal(filtered.provider_name, "openai")

  assert.deepEqual(usageRollupSearchWithoutFilters(filtered), {
    range: "custom",
    start_at: "2026-07-01T00:00:00Z",
    end_at: "2026-07-02T00:00:00Z",
    agent_name: undefined,
    provider_name: undefined,
    model: undefined,
    environment_name: undefined,
    parent_session_id: undefined,
    causal_budget_id: undefined,
    label: undefined,
    label_selector: undefined,
  })
})

test("filter helpers keep bounded drill-down links aligned with aggregate scope", () => {
  const search = validateUsageRollupSearch({
    agent_name: "operator",
    provider_name: "openai",
    label: ["tenant=acme"],
    label_selector: ["!archived"],
  })
  assert.equal(usageRollupFilterCount(search), 4)
  assert.deepEqual(multilineUsageFilters(" tenant=acme\n\n tier=critical "), [
    "tenant=acme",
    "tier=critical",
  ])
  assert.deepEqual(usageRollupSessionSearch(search), {
    agent_name: "operator",
    provider_name: "openai",
    model: undefined,
    environment_name: undefined,
    parent_session_id: undefined,
    causal_budget_id: undefined,
    label: ["tenant=acme"],
    label_selector: ["!archived"],
  })
})
