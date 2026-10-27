import assert from "node:assert/strict"
import test from "node:test"
import { newStructuredJudgeAssertion } from "../src/lib/eval-judge-authoring.ts"
import {
  authoredSuiteCostBudgetContract,
  evalSuiteDraftFromDocument,
  validateEvalSuiteDraft,
} from "../src/lib/eval-suite-authoring.ts"
import {
  createPromotionAssertion,
  PROMOTION_ASSERTION_KINDS,
} from "../src/lib/evaluation-promotion.ts"

const revision = (digit) => `sha256:${digit.repeat(64)}`

test("all twenty authored check types survive a loaded revision and unrelated edits", () => {
  const checks = PROMOTION_ASSERTION_KINDS.map((kind) => createPromotionAssertion(kind, []))
  checks.find((check) => check.kind === "final_output_equals").comparison = "json"
  checks.find((check) => check.kind === "final_output_equals").expected =
    '{"enabled": true, "amount": 1}'
  checks.find((check) => check.kind === "tools_called_in_order").tool_names = [
    "read",
    "write",
    "read",
  ]
  checks.find((check) => check.kind === "tool_called").min_count = 0
  checks.find((check) => check.kind === "tool_called").max_count = 0
  checks.push({
    id: "plain-judge",
    kind: "model_judge",
    evaluator_key: "quality",
    rubric: "Explain the evidence",
    rubric_version: "v1",
    threshold: 0.75,
    include_transcript: true,
  })
  const structured = newStructuredJudgeAssertion(
    { key: "quality", revision: revision("a") },
    checks.map((check) => check.id),
  )
  structured.rubric.revision = revision("b")
  checks.push(structured)
  assert.equal(checks.length, 20)
  assert.equal(new Set(checks.map((check) => check.kind)).size, 20)
  const document = {
    schema_version: 3,
    revision: revision("c"),
    target_key: "assistant.default",
    suite: {
      id: "all-checks",
      revision: revision("d"),
      name: "All checks",
      trial_request: {
        trials: 3,
        timeout_seconds: 120,
        trial_policy: { minimum_passed_trials: 2, max_concurrency: 2 },
      },
    },
    cases: [
      {
        revision: revision("e"),
        id: "case-one",
        name: "Case one",
        source: null,
        stimulus: {
          kind: "simple_input",
          input: { messages: [{ role: "user", text: "Find the answer" }] },
        },
        assertions: checks,
      },
    ],
  }
  const before = structuredClone(document)
  const draft = evalSuiteDraftFromDocument(document)
  draft.name = "Renamed suite"
  const validated = validateEvalSuiteDraft(draft)
  assert.equal(validated.ok, true, validated.error)
  assert.deepEqual(draft.trial_request, {
    trials: 3,
    timeout_seconds: 120,
    minimum_passed_trials: 2,
    max_concurrency: 2,
  })
  assert.deepEqual(draft.cases[0].assertions.slice(0, 19), checks.slice(0, 19))
  const { revision: _, ...rubric } = structured.rubric
  assert.deepEqual(draft.cases[0].assertions[19], { ...structured, rubric })
  assert.deepEqual(document, before)
})

test("exact launch recovery retains all admission fields and fences late settlement", async () => {
  const { saveSuiteLaunchAttempt, readSuiteLaunchAttempt, clearSuiteLaunchAttempt } = await import(
    "../src/lib/eval-suite-launch-attempt.ts"
  )
  const entries = new Map()
  const storage = {
    getItem: (key) => entries.get(key) ?? null,
    setItem: (key, value) => entries.set(key, value),
    removeItem: (key) => entries.delete(key),
  }
  const old = {
    key: "cayu-dashboard-11111111-1111-4111-8111-111111111111",
    revision: revision("a"),
    request: {
      case_ids: ["selected-only"],
      cost_budget: { max_estimated_cost: "0.125", currency: "EUR" },
      expected_execution_profiles: [
        { case_ids: ["selected-only"], execution_profile_revision: revision("b") },
      ],
      expected_exposure_revision: revision("c"),
    },
  }
  saveSuiteLaunchAttempt(storage, "/api", old)
  assert.deepEqual(readSuiteLaunchAttempt(storage, "/api", old.revision), old)
  assert.equal(readSuiteLaunchAttempt(storage, "/other-api", old.revision), null)
  const newer = structuredClone(old)
  newer.key = "cayu-dashboard-22222222-2222-4222-8222-222222222222"
  newer.request.cost_budget.currency = "USD"
  assert.throws(() => saveSuiteLaunchAttempt(storage, "/api", newer), /Resolve/)
  assert.equal(clearSuiteLaunchAttempt(storage, "/api", old), true)
  saveSuiteLaunchAttempt(storage, "/api", newer)
  assert.equal(clearSuiteLaunchAttempt(storage, "/api", old), false)
  assert.deepEqual(readSuiteLaunchAttempt(storage, "/api", old.revision), newer)
  for (const key of entries.keys()) entries.set(key, '{"request":true}')
  assert.throws(() => readSuiteLaunchAttempt(storage, "/api", old.revision), /cannot be read/)
})

test("published currency identifiers survive exact launch recovery", async () => {
  const { saveSuiteLaunchAttempt, readSuiteLaunchAttempt } = await import(
    "../src/lib/eval-suite-launch-attempt.ts"
  )
  for (const currency of ["USD", "A", "USDC.E", "USD_CREDIT", "X-2", "A123456789012345"]) {
    const entries = new Map()
    const storage = {
      getItem: (key) => entries.get(key) ?? null,
      setItem: (key, value) => entries.set(key, value),
      removeItem: (key) => entries.delete(key),
    }
    const budget = authoredSuiteCostBudgetContract("1", currency, {
      cost_budget_available: true,
      cost_budget_currencies: [currency],
    })
    assert.ok(budget)
    const attempt = {
      key: "cayu-dashboard-11111111-1111-4111-8111-111111111111",
      revision: revision("a"),
      request: {
        cost_budget: budget,
        expected_execution_profiles: [
          { case_ids: ["one"], execution_profile_revision: revision("b") },
        ],
        expected_exposure_revision: revision("c"),
      },
    }
    saveSuiteLaunchAttempt(storage, "/api", attempt)
    assert.deepEqual(readSuiteLaunchAttempt(storage, "/api", attempt.revision), attempt)
  }
})
