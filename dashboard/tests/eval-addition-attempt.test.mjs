import assert from "node:assert/strict"
import test from "node:test"
import {
  clearEvalAdditionAttempt,
  readEvalAdditionAttempt,
  saveEvalAdditionAttempt,
} from "../src/lib/eval-addition-attempt.ts"

const digest = (char) => `sha256:${char.repeat(64)}`
const attempt = {
  key: "cayu-dashboard-11111111-2222-4333-8444-555555555555",
  request: {
    mode: "score",
    destination: { kind: "existing", selector: "exact-suite" },
    case_id: "captured",
    case_name: "Captured case",
    assertions: [{ kind: "root_status", id: "complete", expected: "completed" }],
    expected_plan_revision: digest("a"),
    expected_source_revision: digest("b"),
  },
  plan: {
    revision: digest("a"),
    source_revision: digest("b"),
    source_binding_revision: digest("c"),
    mode: "score",
    case_id: "captured",
    case_name: "Captured case",
    case_revision: digest("d"),
    case_definition: { id: "captured", revision: digest("d") },
    destination_case_count: 2,
    membership_changed: false,
    execution_case_count: 1,
    captured_result_revision: digest("e"),
    destination: {
      source: "authored",
      target_key: "target",
      revision: digest("f"),
      suite_id: null,
    },
  },
}
function storage() {
  const rows = new Map()
  return {
    rows,
    getItem: (key) => rows.get(key) ?? null,
    setItem: (key, value) => rows.set(key, value),
    removeItem: (key) => rows.delete(key),
  }
}
test("reload recovery retains the entire reviewed request and same key after acknowledgement loss", () => {
  const store = storage()
  saveEvalAdditionAttempt(store, "/api", "session", attempt)
  const recovered = readEvalAdditionAttempt(store, "/api", "session")
  assert.deepEqual(recovered, attempt)
  assert.equal(readEvalAdditionAttempt(store, "/other-api", "session"), null)
  assert.equal(readEvalAdditionAttempt(store, "/api", "other-session"), null)
  recovered.request.case_name = "Changed in memory"
  assert.deepEqual(readEvalAdditionAttempt(store, "/api", "session"), attempt)
  clearEvalAdditionAttempt(store, "/api", "session", attempt)
  assert.equal(readEvalAdditionAttempt(store, "/api", "session"), null)
})
test("malformed or mismatched recovery material cannot silently become another operation", () => {
  const store = storage()
  saveEvalAdditionAttempt(store, "/api", "session", attempt)
  const key = [...store.rows.keys()][0]
  for (const value of [
    "{",
    "null",
    JSON.stringify({ ...attempt, key: true }),
    JSON.stringify({ ...attempt, plan: { ...attempt.plan, revision: digest("c") } }),
  ]) {
    store.setItem(key, value)
    assert.throws(
      () => readEvalAdditionAttempt(store, "/api", "session"),
      /saved addition attempt cannot be read/,
    )
  }
})
test("a recovery storage failure propagates before a publication can be dispatched", () => {
  const store = {
    ...storage(),
    setItem: () => {
      throw new Error("quota")
    },
  }
  assert.throws(() => saveEvalAdditionAttempt(store, "/api", "session", attempt), /quota/)
})
test("HTTP addition publication carries the exact reviewed tuple and required idempotency key", async () => {
  globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }
  const api = await import("../src/next/api/eval-additions.ts")
  const original = globalThis.fetch
  const calls = []
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init })
    return new Response("{}")
  }
  try {
    await api.publishEvalAddition("source/id", attempt.request, attempt.key)
    assert.equal(calls[0].url, "/api/evals/sessions/source%2Fid/additions")
    assert.deepEqual(JSON.parse(calls[0].init.body), attempt.request)
    assert.equal(calls[0].init.headers.get("Idempotency-Key"), attempt.key)
    await api.fetchEvalCaseSource("suite/id", "case/id", digest("c"))
    assert.match(calls[1].url, /suite%2Fid\/cases\/case%2Fid\/source\?result_revision=/)
  } finally {
    globalThis.fetch = original
  }
})
test("settlement and abandonment only clear the exact owned recovery tuple", () => {
  const store = storage()
  const newer = structuredClone(attempt)
  newer.key = "cayu-dashboard-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
  saveEvalAdditionAttempt(store, "/api", "session", newer)
  assert.equal(clearEvalAdditionAttempt(store, "/api", "session", attempt), false)
  assert.deepEqual(readEvalAdditionAttempt(store, "/api", "session"), newer)
  const changedBody = structuredClone(newer)
  changedBody.request.case_name = "Different exact request"
  assert.equal(clearEvalAdditionAttempt(store, "/api", "session", changedBody), false)
  assert.deepEqual(readEvalAdditionAttempt(store, "/api", "session"), newer)
  assert.equal(clearEvalAdditionAttempt(store, "/api", "session", newer), true)
  assert.equal(readEvalAdditionAttempt(store, "/api", "session"), null)
})
