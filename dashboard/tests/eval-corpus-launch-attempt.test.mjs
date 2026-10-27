import assert from "node:assert/strict"
import test from "node:test"
import {
  clearCorpusLaunchAttempt,
  readCorpusLaunchAttempt,
  saveCorpusLaunchAttempt,
} from "../src/lib/eval-corpus-launch-attempt.ts"

const revision = (digit) => `sha256:${digit.repeat(64)}`
const attempt = {
  key: "cayu-dashboard-12345678-1234-1234-1234-123456789abc",
  request: {
    corpus_revision: revision("a"),
    suite_id: "corpus-suite",
    max_concurrency: 2,
    expected_execution_profile_revision: revision("b"),
  },
}
const storage = () => {
  const values = new Map()
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  }
}
test("corpus recovery restores the complete admission envelope independently of live metadata", () => {
  const store = storage()
  saveCorpusLaunchAttempt(store, "/api", "suite-selector", attempt)
  assert.deepEqual(readCorpusLaunchAttempt(store, "/api", "suite-selector"), attempt)
  assert.equal(readCorpusLaunchAttempt(store, "/different-api", "suite-selector"), null)
  assert.equal(readCorpusLaunchAttempt(store, "/api", "different-suite"), null)
  const changed = structuredClone(attempt)
  changed.request.expected_execution_profile_revision = revision("c")
  assert.throws(() => saveCorpusLaunchAttempt(store, "/api", "suite-selector", changed))
  assert.equal(clearCorpusLaunchAttempt(store, "/api", "suite-selector", changed), false)
  assert.deepEqual(readCorpusLaunchAttempt(store, "/api", "suite-selector"), attempt)
  assert.equal(clearCorpusLaunchAttempt(store, "/api", "suite-selector", attempt), true)
})
test("corrupt or unsupported recovery bodies fail closed instead of starting another run", () => {
  for (const corrupt of [
    { ...attempt, request: { ...attempt.request, max_concurrency: true } },
    { ...attempt, request: { ...attempt.request, expected_execution_profile_revision: "bad" } },
    { ...attempt, request: { ...attempt.request, limits: { max_total_tokens: 10 } } },
    { ...attempt, key: "unexpected" },
  ]) {
    const store = { getItem: () => JSON.stringify(corrupt) }
    assert.throws(() => readCorpusLaunchAttempt(store, "/api", "suite-selector"), /cannot be read/)
  }
})
