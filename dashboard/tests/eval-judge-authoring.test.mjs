import assert from "node:assert/strict"
import test from "node:test"

import {
  judgeProfileForAssertion,
  judgeRouteForAssertion,
  newJudgeCalibrationDraft,
  newMemoryUseJudgeAssertion,
  newStructuredJudgeAssertion,
  structuredAssertionFromReviewedSuite,
  validateJudgeCalibrationDraft,
  validateStructuredJudgeAssertion,
} from "../src/lib/eval-judge-authoring.ts"

const revision = (digit) => `sha256:${digit.repeat(64)}`

function profile() {
  return {
    schema_version: 1,
    key: "quality-judge",
    revision: revision("a"),
    label: "Quality judge",
    provider_name: "openai",
    model: "gpt-5",
    implementation_revision: revision("b"),
    privacy_policy_key: "public-eval",
    privacy_policy_revision: revision("c"),
    allowed_evidence: ["final_output", "transcript", "public_reference"],
    max_input_tokens: 4_000,
    max_output_tokens: 1_000,
    max_total_tokens: 5_000,
    timeout_seconds: 60,
    same_model_use: "allowed_and_labeled",
  }
}

test("structured judge helpers create revision-free, locally valid rubric material", () => {
  const assertion = newStructuredJudgeAssertion(profile(), ["quality"])

  assert.equal(assertion.id, "quality-2")
  assert.equal("revision" in assertion.rubric, false)
  assert.doesNotThrow(() => validateStructuredJudgeAssertion(assertion))

  assertion.rubric.criteria = [
    {
      id: "correctness",
      name: "Correctness",
      description: "Correct result.",
      weight: "0.7",
    },
    {
      id: "completeness",
      name: "Completeness",
      description: "Complete result.",
      weight: "0.3",
    },
  ]
  assert.doesNotThrow(() => validateStructuredJudgeAssertion(assertion))

  assertion.rubric.criteria[1].weight = "0.29"
  assert.throws(() => validateStructuredJudgeAssertion(assertion), /sum exactly to 1/)
})

test("memory-use judge starts with an explicit trusted-reference requirement", () => {
  const assertion = newMemoryUseJudgeAssertion(profile(), ["memory-use"])

  assert.equal(assertion.id, "memory-use-2")
  assert.equal(assertion.rubric.id, "memory-use")
  assert.deepEqual(
    assertion.rubric.criteria.map((criterion) => criterion.weight),
    ["0.5", "0.3", "0.2"],
  )
  assert.throws(() => validateStructuredJudgeAssertion(assertion), /Expected fact 1/)

  assertion.reference.expected_facts = ["The customer is enrolled in the premium plan."]
  assert.doesNotThrow(() => validateStructuredJudgeAssertion(assertion))

  const incompatibleProfile = { ...profile(), allowed_evidence: ["final_output"] }
  assert.throws(
    () => newMemoryUseJudgeAssertion(incompatibleProfile, []),
    /must permit final-output and public-reference evidence/,
  )
})

test("reviewed assertion matching requires a compiled rubric revision", () => {
  const draft = newStructuredJudgeAssertion(profile(), [])
  const compiled = {
    ...structuredClone(draft),
    rubric: { ...structuredClone(draft.rubric), revision: revision("d") },
  }

  assert.equal(structuredAssertionFromReviewedSuite(draft, [draft]), null)
  assert.equal(structuredAssertionFromReviewedSuite(draft, [compiled]), compiled)
})

test("calibration drafts stay bound to reviewed criteria and fixed evidence", () => {
  const draftAssertion = newStructuredJudgeAssertion(profile(), [])
  const assertion = {
    ...structuredClone(draftAssertion),
    rubric: { ...structuredClone(draftAssertion.rubric), revision: revision("d") },
  }
  const draft = newJudgeCalibrationDraft("assistant.default", assertion, "Answer the question.")

  assert.equal(draft.human_criteria[0].criterion_id, "correctness")
  assert.equal(draft.evidence_source_id, "quality-known-evidence")
  assert.equal(draft.trials, 3)
  assert.doesNotThrow(() => validateJudgeCalibrationDraft(draft))

  draft.human_criteria[0].criterion_id = "different"
  assert.throws(() => validateJudgeCalibrationDraft(draft), /criterion order/)

  const invalidSource = newJudgeCalibrationDraft(
    "assistant.default",
    assertion,
    "Answer the question.",
  )
  invalidSource.evidence_source_id = "not portable"
  assert.throws(() => validateJudgeCalibrationDraft(invalidSource), /Evidence source ID/)
})

test("judge lookup is bound to both profile key and immutable revision", () => {
  const assertion = newStructuredJudgeAssertion(profile(), [])
  const target = {
    judge_profiles: [profile()],
    judge_profile_routes: [
      {
        judge_profile_key: "quality-judge",
        judge_profile_revision: revision("a"),
        candidate_route_relation: "same_model",
      },
    ],
  }

  assert.equal(judgeProfileForAssertion(target, assertion)?.key, "quality-judge")
  assert.equal(judgeRouteForAssertion(target, assertion), "same_model")
  assertion.judge_profile_revision = revision("f")
  assert.equal(judgeProfileForAssertion(target, assertion), undefined)
  assert.equal(judgeRouteForAssertion(target, assertion), undefined)
})
