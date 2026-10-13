import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"

import { newEvalSuiteDraft, validateEvalSuiteDraft } from "../src/lib/eval-suite-authoring.ts"
import {
  validateCapturedEvaluationDraft,
  validatePromotionDraft,
} from "../src/lib/evaluation-promotion.ts"
import { EVAL_TEXT_LIMITS } from "../src/lib/generated/eval-text-limits.ts"

const cases = JSON.parse(
  readFileSync(new URL("./fixtures/eval-text-boundaries.json", import.meta.url)),
)

test("generated evaluation limits declare the preserved scalar contract", () => {
  assert.deepEqual(EVAL_TEXT_LIMITS, {
    unit: "unicode_scalar_values",
    name: 256,
    description: 2048,
  })
})

for (const boundary of cases) {
  test(`evaluation editors preserve shared boundary: ${boundary.id}`, () => {
    const value = boundary.text === null ? null : boundary.text.repeat(boundary.repeat)
    for (const target of ["suite", "case"]) {
      const authored = newEvalSuiteDraft("target")
      const record = target === "suite" ? authored : authored.cases[0]
      record[boundary.field] = value
      assert.equal(validateEvalSuiteDraft(authored).ok, boundary.accepted, `authored ${target}`)

      const promotion = {
        expected_baseline_revision: `sha256:${"a".repeat(64)}`,
        suite: {
          id: authored.id,
          name: authored.name,
          description: authored.description,
          trial_request: { trials: 1, timeout_seconds: 60 },
        },
        case: {
          id: authored.cases[0].id,
          suite_id: authored.id,
          name: authored.cases[0].name,
          description: authored.cases[0].description,
          input: { messages: [{ role: "user", text: "Run the example." }] },
          assertions: [{ id: "completed", kind: "root_status", expected: "completed" }],
        },
      }
      assert.equal(validatePromotionDraft(promotion).ok, boundary.accepted, `promotion ${target}`)
      const captured = structuredClone(promotion)
      delete captured.suite.trial_request
      delete captured.case.input
      assert.equal(
        validateCapturedEvaluationDraft(captured).ok,
        boundary.accepted,
        `captured ${target}`,
      )
    }
  })
}
