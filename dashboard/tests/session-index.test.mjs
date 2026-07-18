import assert from "node:assert/strict"
import test from "node:test"
import { parseDashboardSearch, stringifyDashboardSearch } from "../src/lib/search-params.ts"
import {
  DEFAULT_SESSION_ORDER,
  multilineSessionFilters,
  sessionIndexAdvancedFilterKey,
  sessionIndexFilterCount,
  sessionIndexQuery,
  sessionIndexSearchWithCursor,
  sessionIndexSearchWithFilters,
  sessionIndexSearchWithoutFilters,
  validateSessionIndexSearch,
} from "../src/lib/session-index-search.ts"

test("dashboard URL parsing preserves JSON-like scalar identifiers as strings", () => {
  const parsed = parseDashboardSearch(
    "?q=123&agent_name=true&environment_name=null&model=1e3&parent_session_id=%22quoted%22&label=tenant%3Dacme&label=tier%3Dcritical&label_selector=true&label_selector=%21archived",
  )

  assert.deepEqual(validateSessionIndexSearch(parsed), {
    q: "123",
    status: undefined,
    debug_state: undefined,
    agent_name: "true",
    provider_name: undefined,
    model: "1e3",
    environment_name: "null",
    parent_session_id: '"quoted"',
    causal_budget_id: undefined,
    order_by: undefined,
    label: ["tenant=acme", "tier=critical"],
    label_selector: ["true", "!archived"],
    cursor: undefined,
  })
})

test("dashboard URL serialization round-trips repeated values without JSON coercion", () => {
  const search = validateSessionIndexSearch({
    q: "123",
    agent_name: "true",
    environment_name: "null",
    label: ["tenant=acme", "tier=critical"],
    label_selector: ["true", "!archived"],
  })

  const serialized = stringifyDashboardSearch(search)
  const params = new URLSearchParams(serialized)
  assert.deepEqual(params.getAll("label"), ["tenant=acme", "tier=critical"])
  assert.deepEqual(params.getAll("label_selector"), ["true", "!archived"])
  assert.deepEqual(validateSessionIndexSearch(parseDashboardSearch(serialized)), search)
})

test("session-index search validation canonicalizes supported URL state", () => {
  assert.deepEqual(
    validateSessionIndexSearch({
      q: "  customer failure  ",
      status: "failed",
      debug_state: "tool_issue",
      agent_name: " builder ",
      provider_name: "openai",
      model: " gpt-example ",
      environment_name: " production ",
      parent_session_id: "parent-1",
      causal_budget_id: " budget-1 ",
      order_by: "created_at_asc",
      label: [" tenant=acme ", " tier=critical "],
      label_selector: ["region in (us,eu)", " !archived "],
      cursor: " cursor-2 ",
      unrelated: "ignored",
    }),
    {
      q: "customer failure",
      status: "failed",
      debug_state: "tool_issue",
      agent_name: "builder",
      provider_name: "openai",
      model: "gpt-example",
      environment_name: "production",
      parent_session_id: "parent-1",
      causal_budget_id: "budget-1",
      order_by: "created_at_asc",
      label: ["tenant=acme", "tier=critical"],
      label_selector: ["region in (us,eu)", "!archived"],
      cursor: "cursor-2",
    },
  )
})

test("invalid, empty, and default URL values are omitted", () => {
  assert.deepEqual(
    validateSessionIndexSearch({
      q: " ",
      status: "unknown",
      debug_state: ["tool_issue"],
      order_by: DEFAULT_SESSION_ORDER,
      label: ["", "  "],
      label_selector: 42,
      cursor: null,
    }),
    {
      q: undefined,
      status: undefined,
      debug_state: undefined,
      agent_name: undefined,
      provider_name: undefined,
      model: undefined,
      environment_name: undefined,
      parent_session_id: undefined,
      causal_budget_id: undefined,
      order_by: undefined,
      label: undefined,
      label_selector: undefined,
      cursor: undefined,
    },
  )
})

test("the generated summary query receives every server-side filter", () => {
  const search = validateSessionIndexSearch({
    q: "incident",
    status: "interrupted",
    debug_state: "interruption",
    agent_name: "operator",
    provider_name: "anthropic",
    model: "claude-example",
    environment_name: "prod",
    parent_session_id: "parent",
    causal_budget_id: "budget",
    order_by: "updated_at_asc",
    label: ["tenant=acme"],
    label_selector: ["region in (us,eu)"],
    cursor: "opaque-cursor",
  })

  assert.deepEqual(sessionIndexQuery(search), {
    limit: 100,
    cursor: "opaque-cursor",
    q: "incident",
    status: "interrupted",
    debug_state: "interruption",
    agent_name: "operator",
    provider_name: "anthropic",
    model: "claude-example",
    environment_name: "prod",
    parent_session_id: "parent",
    causal_budget_id: "budget",
    order_by: "updated_at_asc",
    label: ["tenant=acme"],
    label_selector: ["region in (us,eu)"],
  })
})

test("result-shaping changes reset the opaque cursor atomically", () => {
  const current = validateSessionIndexSearch({
    q: "old",
    status: "running",
    cursor: "page-3",
  })

  assert.deepEqual(sessionIndexSearchWithFilters(current, { q: " new " }), {
    q: "new",
    status: "running",
    debug_state: undefined,
    agent_name: undefined,
    provider_name: undefined,
    model: undefined,
    environment_name: undefined,
    parent_session_id: undefined,
    causal_budget_id: undefined,
    order_by: undefined,
    label: undefined,
    label_selector: undefined,
    cursor: undefined,
  })
  assert.equal(sessionIndexSearchWithCursor(current, " next-page ").cursor, "next-page")
})

test("clearing filters preserves ordering while removing filters and pagination", () => {
  const current = validateSessionIndexSearch({
    q: "failure",
    status: "failed",
    agent_name: "builder",
    order_by: "created_at_asc",
    cursor: "page-3",
  })

  assert.deepEqual(sessionIndexSearchWithoutFilters(current), {
    q: undefined,
    status: undefined,
    debug_state: undefined,
    agent_name: undefined,
    provider_name: undefined,
    model: undefined,
    environment_name: undefined,
    parent_session_id: undefined,
    causal_budget_id: undefined,
    order_by: "created_at_asc",
    label: undefined,
    label_selector: undefined,
    cursor: undefined,
  })
})

test("multiline filters preserve selector commas and repeated URL values", () => {
  assert.deepEqual(multilineSessionFilters(" tenant=acme\n\nregion in (us, eu)\r\n !archived "), [
    "tenant=acme",
    "region in (us, eu)",
    "!archived",
  ])
  assert.equal(multilineSessionFilters(" \n "), undefined)
})

test("filter counts and advanced keys retain every active criterion", () => {
  const first = validateSessionIndexSearch({
    q: "failure",
    agent_name: "builder",
    label: ["tenant=acme", "tier=critical"],
    label_selector: ["!archived"],
  })
  const second = validateSessionIndexSearch({
    ...first,
    label: ["tenant=other", "tier=critical"],
  })

  assert.equal(sessionIndexFilterCount(first), 5)
  assert.notEqual(sessionIndexAdvancedFilterKey(first), sessionIndexAdvancedFilterKey(second))
})
