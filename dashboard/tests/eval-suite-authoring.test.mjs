import assert from "node:assert/strict"
import test from "node:test"

import {
  authoredSuiteRunPreviewIdentity,
  blankEvalScenarioDraft,
  duplicateEvalSuiteCase,
  EVAL_SUITE_MAX_CASES,
  evalSuiteDraftFromDocument,
  evalSuitePreviewMatchesDraft,
  newEvalSuiteDraft,
  newSimpleEvalCase,
  scenarioArtifactBindingsRequireMaterialization,
  validateEvalSuiteDraft,
} from "../src/lib/eval-suite-authoring.ts"

const revision = (digit) => `sha256:${digit.repeat(64)}`

function authoredDocument() {
  return {
    schema_version: 1,
    revision: revision("a"),
    target_key: "assistant.default",
    suite: {
      id: "support-regressions",
      revision: revision("b"),
      name: "Support regressions",
      description: "Reusable support behavior",
      trial_request: { trials: 1, timeout_seconds: 120 },
    },
    cases: [
      {
        revision: revision("c"),
        id: "refund-case",
        name: "Refund case",
        description: null,
        source: null,
        stimulus: {
          kind: "simple_input",
          input: { messages: [{ role: "user", text: "Refund invoice 123." }] },
        },
        assertions: [{ id: "completed", kind: "root_status", expected: "completed" }],
      },
    ],
  }
}

test("new suite and case helpers create complete deterministic drafts", () => {
  const draft = newEvalSuiteDraft("assistant.default")
  assert.equal(draft.target_key, "assistant.default")
  assert.equal(draft.trial_request.trials, 1)
  assert.deepEqual(validateEvalSuiteDraft(draft), { ok: true, draft })

  const added = newSimpleEvalCase(draft.cases)
  assert.equal(added.id, "case")
  assert.deepEqual(added.assertions, [
    { id: "root_status", kind: "root_status", expected: "completed" },
  ])

  const duplicate = duplicateEvalSuiteCase([draft.cases[0], added], 0)
  assert.equal(duplicate.id, "case-1-copy")
  duplicate.assertions[0].id = "changed"
  assert.equal(draft.cases[0].assertions[0].id, "root_status")
})

test("saved suite projection is editable without mutating immutable server material", () => {
  const document = authoredDocument()
  const draft = evalSuiteDraftFromDocument(document)

  assert.equal(evalSuitePreviewMatchesDraft(document, draft), true)
  assert.equal("revision" in draft.cases[0], false)
  draft.cases[0].stimulus.input.messages[0].text = "Changed"
  assert.equal(document.cases[0].stimulus.input.messages[0].text, "Refund invoice 123.")
  assert.equal(evalSuitePreviewMatchesDraft(document, draft), false)
})

test("launch preview identity is bound to the exact saved and reviewed suite revision", () => {
  const selection = { case_ids: ["refund-case"] }
  const current = authoredSuiteRunPreviewIdentity(revision("a"), revision("a"), selection)

  assert.equal(current, authoredSuiteRunPreviewIdentity(revision("a"), revision("a"), selection))
  assert.notEqual(current, authoredSuiteRunPreviewIdentity(revision("a"), revision("b"), selection))
})

test("only nonblank embedded artifact overrides require fixture materialization", () => {
  assert.equal(scenarioArtifactBindingsRequireMaterialization({}, true), false)
  assert.equal(
    scenarioArtifactBindingsRequireMaterialization({ "input-file": "  \t" }, true),
    false,
  )
  assert.equal(
    scenarioArtifactBindingsRequireMaterialization({ "input-file": "retained-artifact-123" }, true),
    true,
  )
  assert.equal(
    scenarioArtifactBindingsRequireMaterialization(
      { "input-file": "retained-artifact-123" },
      false,
    ),
    false,
  )
})

test("blank scenario authoring is target-bound and starts with one initial input", () => {
  const evalCase = newEvalSuiteDraft("assistant.default").cases[0]
  const scenario = blankEvalScenarioDraft("assistant.default", evalCase)

  assert.equal(scenario.id, "case-1-scenario")
  assert.equal(scenario.target_key, "assistant.default")
  assert.equal(scenario.events.length, 1)
  assert.equal(scenario.events[0].kind, "initial")
  assert.equal(scenario.events[0].sequence, 0)
})

test("suite validation mirrors Python Unicode whitespace and immutable scenario references", () => {
  const blank = newEvalSuiteDraft("assistant.default")
  blank.cases[0].stimulus.input.messages[0].text = "\u0085"
  assert.match(validateEvalSuiteDraft(blank).error, /must contain/)

  blank.cases[0].stimulus.input.messages[0].text = "\ufeff"
  assert.deepEqual(validateEvalSuiteDraft(blank), { ok: true, draft: blank })

  const outerWhitespace = newEvalSuiteDraft("assistant.default")
  outerWhitespace.name = "\u0085Evaluation suite"
  assert.match(validateEvalSuiteDraft(outerWhitespace).error, /without outer whitespace/)

  const scenario = newEvalSuiteDraft("assistant.default")
  scenario.cases[0].stimulus = {
    kind: "scenario",
    scenario_id: "saved-scenario",
    scenario_revision: "pending",
  }
  assert.match(validateEvalSuiteDraft(scenario).error, /saved immutable scenario revision/)

  const opaque = newEvalSuiteDraft("external.candidate")
  opaque.cases[0].stimulus.input = {
    opaque_external_case_ref: { id: "private-case", revision: revision("d") },
  }
  assert.match(validateEvalSuiteDraft(opaque).error, /runtime-owned opaque external input/)
})

test("suite validation enforces the server's 1,000-case ceiling", () => {
  const draft = newEvalSuiteDraft("assistant.default")
  draft.cases = Array.from({ length: EVAL_SUITE_MAX_CASES + 1 }, () =>
    structuredClone(draft.cases[0]),
  )

  assert.match(validateEvalSuiteDraft(draft).error, /between 1 and 1,000 cases/)
  assert.throws(
    () => newSimpleEvalCase(draft.cases.slice(0, EVAL_SUITE_MAX_CASES)),
    /cannot contain more than 1000 cases/,
  )
  assert.throws(
    () => duplicateEvalSuiteCase(draft.cases.slice(0, EVAL_SUITE_MAX_CASES), 0),
    /cannot contain more than 1000 cases/,
  )
})
