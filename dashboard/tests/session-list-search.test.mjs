import assert from "node:assert/strict"
import test from "node:test"
import {
  sessionIndexSearchWithCursor,
  sessionIndexSearchWithFilters,
  sessionOverviewQuery,
  validateSessionIndexSearch,
} from "../src/lib/session-index-search.ts"

test("overview query preserves every old filter and excludes evals and forks by default", () => {
  const source = {
    q: "report",
    status: "failed",
    debug_state: "tool_issue",
    agent_name: "reviewer",
    provider_name: "provider",
    model: "model",
    environment_name: "local",
    parent_session_id: "parent",
    causal_budget_id: "budget",
    label: ["team=a"],
    label_selector: ["tier in (a,b)", "!hidden"],
    order_by: "created_at_asc",
    cursor: "opaque",
    state: "waiting",
  }
  const query = sessionOverviewQuery(validateSessionIndexSearch(source))
  assert.deepEqual(query, {
    ...Object.fromEntries(Object.entries(source).filter(([key]) => key !== "state")),
    limit: 100,
    include: ["overview"],
    overview_state: ["waiting_approval", "waiting_input", "needs_recovery"],
    include_eval: false,
    include_forks: false,
  })
})

test("URL toggles, individual merged states, filter changes and cursor links remain bound", () => {
  const source = validateSessionIndexSearch({
    include_eval: "true",
    include_forks: true,
    state: "failed",
    cursor: "old",
    agent_name: "a",
  })
  const query = sessionOverviewQuery(source)
  assert.equal(query.include_eval, true)
  assert.equal(query.include_forks, true)
  assert.deepEqual(query.overview_state, ["failed"])
  const changed = sessionIndexSearchWithFilters(source, { state: "completed" })
  assert.equal(changed.cursor, undefined)
  assert.equal(changed.agent_name, "a")
  assert.equal(changed.include_forks, true)
  assert.deepEqual(sessionIndexSearchWithCursor(changed, "next"), { ...changed, cursor: "next" })
  const invalid = sessionOverviewQuery(
    validateSessionIndexSearch({
      state: "unknown",
      include_eval: "false",
      include_forks: "anything",
    }),
  )
  assert.equal(invalid.overview_state, undefined)
  assert.equal(invalid.include_eval, false)
  assert.equal(invalid.include_forks, false)
})
